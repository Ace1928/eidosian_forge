"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      EIDOSIAN LOGGING SYSTEM                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
Advanced logging with structured output, rotation, and contextual enrichment.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union


class LogLevel(Enum):
    """Log levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def __init__(self, include_traceback: bool = True):
        super().__init__()
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                try:
                    json.dumps(value)  # Test serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        # Add exception info
        if record.exc_info and self.include_traceback:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[2] else None,
            }

        return json.dumps(log_data)


class ColorFormatter(logging.Formatter):
    """Colored console log formatter."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self, fmt: str, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"
        return super().format(record)


class EidosianLogger:
    """
    Enhanced logger with context tracking and structured output.

    Features:
    - Automatic context enrichment
    - Structured JSON output option
    - File rotation
    - Performance tracking
    - Thread-safe context management
    """

    _context = threading.local()

    def __init__(
        self,
        name: str,
        level: Union[str, int, LogLevel] = LogLevel.INFO,
        structured: bool = False,
        file_path: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):
        self.name = name
        self._logger = logging.getLogger(name)

        # Set level
        if isinstance(level, LogLevel):
            level = level.value
        elif isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(level)

        # Clear existing handlers
        self._logger.handlers.clear()

        # Console handler
        # Use stderr for diagnostic logs so stdout remains protocol-safe (e.g., MCP stdio JSON-RPC).
        console_handler = logging.StreamHandler(sys.stderr)
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                ColorFormatter(
                    "[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s", "%Y-%m-%d %H:%M:%S"
                )
            )
        self._logger.addHandler(console_handler)

        # File handler
        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setFormatter(StructuredFormatter())
            self._logger.addHandler(file_handler)

        self._logger.propagate = False

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get current logging context."""
        if not hasattr(cls._context, "data"):
            cls._context.data = {}
        return cls._context.data

    @classmethod
    def set_context(cls, **kwargs) -> None:
        """Add to logging context."""
        ctx = cls.get_context()
        ctx.update(kwargs)

    @classmethod
    def clear_context(cls) -> None:
        """Clear logging context."""
        cls._context.data = {}

    @classmethod
    @contextmanager
    def context(cls, **kwargs):
        """Context manager for temporary context."""
        old_context = cls.get_context().copy()
        cls.set_context(**kwargs)
        try:
            yield
        finally:
            cls._context.data = old_context

    def _log(self, level: int, msg: str, *args, exc_info=None, **kwargs):
        """Internal logging with context enrichment."""
        extra = {**self.get_context(), **kwargs.pop("extra", {})}
        self._logger.log(level, msg, *args, exc_info=exc_info, extra=extra, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, exc_info: bool = False, **kwargs):
        self._log(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args, exc_info: bool = True, **kwargs):
        self._log(logging.CRITICAL, msg, *args, exc_info=exc_info, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, exc_info=True, **kwargs)

    def function_entry(
        self,
        func_name: str,
        args: tuple = (),
        kwargs: dict = None,
        max_length: int = 200,
    ):
        """Log function entry."""
        kwargs = kwargs or {}
        args_str = self._truncate(str(args), max_length)
        kwargs_str = self._truncate(str(kwargs), max_length)
        self.debug(f"→ {func_name}(args={args_str}, kwargs={kwargs_str})")

    def function_exit(
        self,
        func_name: str,
        result: Any = None,
        duration_ms: float = None,
        max_length: int = 500,
    ):
        """Log function exit."""
        result_str = self._truncate(str(result), max_length)
        duration_str = f" [{duration_ms:.2f}ms]" if duration_ms else ""
        self.debug(f"← {func_name} returned {result_str}{duration_str}")

    def function_error(
        self,
        func_name: str,
        error: Exception,
        duration_ms: float = None,
    ):
        """Log function error."""
        duration_str = f" [{duration_ms:.2f}ms]" if duration_ms else ""
        self.error(f"✗ {func_name} raised {type(error).__name__}: {error}{duration_str}", exc_info=True)

    @staticmethod
    def _truncate(s: str, max_length: int) -> str:
        """Truncate string with ellipsis."""
        if len(s) <= max_length:
            return s
        return s[: max_length - 3] + "..."


# Module-level convenience functions
_loggers: Dict[str, EidosianLogger] = {}


def get_logger(name: str = None, level: Union[str, int, LogLevel] = LogLevel.INFO, **kwargs) -> EidosianLogger:
    """Get or create a logger."""
    if name is None:
        name = "eidosian"

    if name not in _loggers:
        _loggers[name] = EidosianLogger(name, level=level, **kwargs)

    return _loggers[name]


def configure_logging(
    level: Union[str, int, LogLevel] = LogLevel.INFO,
    structured: bool = False,
    file_path: Optional[str] = None,
):
    """Configure the root Eidosian logger."""
    return get_logger("eidosian", level=level, structured=structured, file_path=file_path)

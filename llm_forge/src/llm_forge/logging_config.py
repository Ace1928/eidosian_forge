"""
Advanced Logging Configuration System for Eidosian Forge.

This module provides a sophisticated, configurable logging system with
stylized output, contextual awareness, and adaptive behavior. Features
witty prefixes, color-coded output by log level, and flexible handler
configuration.
"""

import inspect
import logging
import random
import sys
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

# Type definitions for logging configuration
LogLevel = Union[int, str]
LoggerType = logging.Logger
FormatStyle = Literal["%", "{", "$"]


class ColorCode(Enum):
    """
    ANSI color and style codes for terminal output enhancement.

    Provides color and styling options for log messages in terminal output.
    """

    GREY = "\033[38;5;240m"
    RED = "\033[31;1m"
    GREEN = "\033[32;1m"
    YELLOW = "\033[33;1m"
    BLUE = "\033[34;1m"
    MAGENTA = "\033[35;1m"
    CYAN = "\033[36;1m"
    WHITE = "\033[37;1m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


class EidosianFormatter(logging.Formatter):
    """
    Enhanced log formatter with color support and witty prefixes.

    Extends the standard logging.Formatter to add color-coding and
    contextual prefixes based on log level.

    Attributes:
        LEVEL_COLORS: Mapping of log levels to terminal color codes
        WITTY_PREFIXES: Collection of witty message prefixes per log level
        use_color: Flag to enable/disable colored output
    """

    LEVEL_COLORS: Dict[int, ColorCode] = {
        logging.DEBUG: ColorCode.BLUE,
        logging.INFO: ColorCode.GREEN,
        logging.WARNING: ColorCode.YELLOW,
        logging.ERROR: ColorCode.RED,
        logging.CRITICAL: ColorCode.MAGENTA,
    }

    WITTY_PREFIXES: Dict[int, List[str]] = {
        logging.DEBUG: [
            "🔍 Elementary, my dear Watson:",
            "🐞 Bug whisperer reports:",
            "🔬 Under the microscope:",
            "🔧 Tinkering in progress:",
        ],
        logging.INFO: [
            "💡 Enlightenment arrived:",
            "📢 Hear ye, hear ye:",
            "🌟 Cosmic insight:",
            "🧠 Neural activity detected:",
        ],
        logging.WARNING: [
            "⚠️ Plot twist ahead:",
            "🚨 Spidey sense tingling:",
            "⚡ Disturbance in the Force:",
            "🌩️ Storm on the horizon:",
        ],
        logging.ERROR: [
            "💥 Houston, we have a problem:",
            "🔥 Code on fire:",
            "🚫 Reality check failed:",
            "💀 Gremlins detected:",
        ],
        logging.CRITICAL: [
            "☢️ Meltdown imminent:",
            "🌋 Volcanic exception erupted:",
            "🧟 Zombie apocalypse initiated:",
            "🚀 Ejection sequence activated:",
        ],
    }

    def __init__(
        self,
        fmt: str,
        datefmt: Optional[str] = None,
        style: FormatStyle = "%",
        use_color: bool = True,
    ) -> None:
        """
        Initialize the Eidosian formatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
            style: Format style ('%', '{', or '$')
            use_color: Whether to apply color to the output
        """
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with color and witty prefixes.

        Adds contextual information, a witty prefix based on log level,
        and applies color coding if enabled.

        Args:
            record: The log record to format

        Returns:
            Formatted log message with styling applied
        """
        # Add contextual info
        record.filepath = record.pathname.split("/")[-1]

        # Add witty prefix
        level_prefixes = self.WITTY_PREFIXES.get(record.levelno, [""])
        record.prefix = random.choice(level_prefixes)

        # Format the message
        formatted_message = super().format(record)

        # Apply color if enabled and output is to terminal
        if self.use_color and sys.stdout.isatty():
            color_code = self.LEVEL_COLORS.get(record.levelno, ColorCode.WHITE)
            return f"{color_code.value}{formatted_message}{ColorCode.RESET.value}"

        return formatted_message


def configure_logging(
    level: LogLevel = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = "%(asctime)s [%(levelname)8s] %(prefix)s %(message)s (%(filepath)s:%(lineno)d)",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    use_color: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    logger_name: Optional[str] = None,
) -> LoggerType:
    """
    Configure and return a customized logger with Eidosian aesthetics.

    Creates a logger with console output and optional file output.
    Automatically detects calling module's name for contextual logging.

    Args:
        level: Logging level (DEBUG, INFO, etc.)
        log_file: Optional path to log file
        log_format: Format string for log messages
        date_format: Format string for timestamps
        use_color: Whether to use color in console output
        max_file_size: Maximum size in bytes for rotating file handler
        backup_count: Number of backup files to keep
        logger_name: Optional explicit logger name

    Returns:
        Configured logger instance ready for use

    Example:
        >>> logger = configure_logging(level="DEBUG", log_file="app.log")
        >>> logger.info("System initialized")
    """
    # Determine appropriate logger name if not provided
    if logger_name is None:
        # Get the calling frame to determine module context
        frame = inspect.currentframe()
        if frame:
            try:
                caller_frame = frame.f_back
                if caller_frame:
                    caller_module = inspect.getmodule(caller_frame)
                    if caller_module:
                        logger_name = caller_module.__name__
            finally:
                del frame

    # Default to root logger if we couldn't determine name
    logger_name = logger_name or "eidosian_forge"

    # Get or create logger
    logger = logging.getLogger(logger_name)

    # Convert string level to numeric if needed
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        level = numeric_level

    # Set level and clear existing handlers
    logger.setLevel(level)
    logger.handlers.clear()

    # Create formatter
    formatter = EidosianFormatter(
        fmt=log_format, datefmt=date_format, style="%", use_color=use_color
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logging
    logger.propagate = False

    # Log a greeting on first configuration
    if not hasattr(configure_logging, "_initialized"):
        logger.debug(
            f"Eidosian logging initialized on {datetime.now().strftime('%Y-%m-%d')} at level {logging.getLevelName(level)}"
        )
        setattr(configure_logging, "_initialized", True)

    return logger

"""
🔹 Enhanced Eidosian Logging Configuration Module 🔹

Sets up a centralized, feature-rich logging configuration for the entire project.

Features:
    • Colorized terminal output with custom symbols
    • Multiple logging handlers (console, file, rotating files, JSON)
    • Rich formatting with contextual information
    • Performance tracking and timing utilities for both sync and async code
    • Structured logging support for analytics
    • Perfect backward compatibility with existing code
    • Module-specific logging control
    • Dynamic log level management
    • Advanced error handling and recovery
"""

import logging
import os
import sys
import time
import json
import platform
import threading
import functools
import datetime
import inspect
import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Set, Tuple, TypeVar, cast

from eidos_config import LOG_LEVEL

# Try to import optional dependencies with graceful fallbacks
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.traceback import install as rich_traceback_install
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# =============================================================================
# Constants and Configuration
# =============================================================================
DEFAULT_LOG_DIR = os.environ.get("EIDOS_LOG_DIR", "logs")
DEFAULT_LOG_FILE = os.environ.get("EIDOS_LOG_FILE", "eidos.log")
DEFAULT_JSON_LOG_FILE = os.environ.get("EIDOS_JSON_LOG_FILE", "eidos_structured.log")
MAX_LOG_SIZE_BYTES = int(os.environ.get("EIDOS_MAX_LOG_SIZE", 10 * 1024 * 1024))  # 10 MB
BACKUP_COUNT = int(os.environ.get("EIDOS_LOG_BACKUP_COUNT", 5))
ENABLE_CONSOLE_COLORS = os.environ.get("EIDOS_CONSOLE_COLORS", "1").lower() in ("1", "true", "yes")
ENABLE_FILE_LOGGING = os.environ.get("EIDOS_FILE_LOGGING", "1").lower() in ("1", "true", "yes")
ENABLE_JSON_LOGGING = os.environ.get("EIDOS_JSON_LOGGING", "0").lower() in ("1", "true", "yes")
ENABLE_EMOJIS = os.environ.get("EIDOS_LOG_EMOJIS", "1").lower() in ("1", "true", "yes")
TRACK_PERFORMANCE = os.environ.get("EIDOS_LOG_PERFORMANCE", "1").lower() in ("1", "true", "yes")

# Custom log format with additional context
DETAILED_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
SIMPLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
JSON_FORMAT = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "file": "%(filename)s", "line": %(lineno)d, "message": "%(message)s"}'
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Emojis for log levels (can be disabled via env var)
LOG_LEVEL_EMOJIS = {
    "DEBUG": "🔍",    # Magnifying glass
    "INFO": "ℹ️",     # Information
    "WARNING": "⚠️",  # Warning
    "ERROR": "❌",    # Error
    "CRITICAL": "🔥", # Fire/critical
}

# Terminal color scheme (used when colorlog is available)
TERMINAL_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

# Flag to track if logging has been initialized
_LOGGING_INITIALIZED = False

# Store module-specific loggers for centralized management
_MODULE_LOGGERS = {}

# =============================================================================
# Configuration Management
# =============================================================================
class LoggingConfig:
    """
    Central configuration store for logging settings.
    Allows dynamic updates and provides defaults.
    """
    def __init__(self):
        self.level = LOG_LEVEL
        self.console_format = DETAILED_FORMAT
        self.file_format = DETAILED_FORMAT
        self.json_format = JSON_FORMAT
        self.date_format = TIME_FORMAT
        self.log_dir = DEFAULT_LOG_DIR
        self.log_file = DEFAULT_LOG_FILE
        self.json_log_file = DEFAULT_JSON_LOG_FILE
        self.max_size = MAX_LOG_SIZE_BYTES
        self.backup_count = BACKUP_COUNT
        self.enable_colors = ENABLE_CONSOLE_COLORS
        self.enable_file = ENABLE_FILE_LOGGING
        self.enable_json = ENABLE_JSON_LOGGING
        self.enable_emojis = ENABLE_EMOJIS
        self.track_performance = TRACK_PERFORMANCE
        self.module_levels = {}  # Module-specific log levels

    def update(self, **kwargs) -> None:
        """Update configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logging.warning(f"Unknown logging configuration parameter: {key}")

    def get_level_for_module(self, module_name: str) -> int:
        """Get the log level for a specific module."""
        # Check for exact match
        if module_name in self.module_levels:
            return self.module_levels[module_name]
        
        # Check for parent module match (e.g. 'eidos.memory' matches 'eidos')
        for prefix, level in self.module_levels.items():
            if module_name.startswith(f"{prefix}."):
                return level
        
        # Fall back to default level
        return self.level

    def set_module_level(self, module_name: str, level: Union[int, str]) -> None:
        """Set log level for a specific module."""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self.module_levels[module_name] = level
        
        # Update existing logger if present
        if module_name in _MODULE_LOGGERS:
            _MODULE_LOGGERS[module_name].setLevel(level)

# Global configuration object
config = LoggingConfig()

# =============================================================================
# Utility Functions
# =============================================================================
def ensure_log_directory() -> str:
    """
    Creates the log directory if it doesn't exist.
    
    Returns:
        str: Path to the log directory.
    """
    log_dir = Path(config.log_dir)
    try:
        log_dir.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        # Fall back to a user-writable directory if we can't create the specified one
        fallback_dir = Path.home() / ".eidos" / "logs"
        fallback_dir.mkdir(exist_ok=True, parents=True)
        logging.warning(f"Could not create log directory {log_dir}, falling back to {fallback_dir}")
        return str(fallback_dir)
    return str(log_dir)

def get_platform_info() -> Dict[str, str]:
    """
    Collects system platform information for log context.
    
    Returns:
        Dict[str, str]: Dictionary with platform details.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "python": platform.python_version(),
        "hostname": platform.node()
    }

def add_emojis_to_format(log_format: str) -> str:
    """
    Adds emoji prefixes to the log format if enabled.
    
    Args:
        log_format (str): Original log format string.
        
    Returns:
        str: Log format with emojis added if enabled.
    """
    if not config.enable_emojis:
        return log_format
        
    # Insert level-specific emojis using a custom formatter
    emoji_format = log_format.replace("[%(levelname)s]", "[%(emoji)s %(levelname)s]")
    return emoji_format

def clean_log_files(max_age_days: int = 30) -> int:
    """
    Clean up old log files beyond the specified age.
    
    Args:
        max_age_days: Maximum age of log files to keep
        
    Returns:
        Number of files removed
    """
    log_dir = Path(config.log_dir)
    if not log_dir.exists():
        return 0
        
    removed = 0
    now = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    try:
        for log_file in log_dir.glob("*.log*"):  # Include rotated logs
            file_age = now - log_file.stat().st_mtime
            if file_age > max_age_seconds:
                log_file.unlink()
                removed += 1
    except Exception as e:
        logging.error(f"Error cleaning log files: {e}")
    
    return removed

def get_log_level_name(level: Union[int, str]) -> str:
    """
    Convert a logging level (int or string) to its canonical name.
    
    Args:
        level: Logging level as int or string
        
    Returns:
        Canonical level name (DEBUG, INFO, etc.)
    """
    if isinstance(level, str):
        return level.upper()
    
    # Convert int level to name
    for name, value in logging._nameToLevel.items():
        if value == level:
            return name
    
    return "INFO"  # Default if unknown

def patch_record_factory():
    """Patch the record factory to add emoji field if missing."""
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        if not hasattr(record, 'emoji'):
            record.emoji = '🔹'  # Default emoji
        return record
    
    logging.setLogRecordFactory(record_factory)

# =============================================================================
# Custom Formatters
# =============================================================================
class EmojiLogFormatter(logging.Formatter):
    """
    Custom log formatter that adds emoji prefixes to log messages.
    """
    def format(self, record: logging.LogRecord) -> str:
        level_name = record.levelname
        if config.enable_emojis and level_name in LOG_LEVEL_EMOJIS:
            record.emoji = LOG_LEVEL_EMOJIS[level_name]
        else:
            record.emoji = ""
        return super().format(record)

class StructuredLogFormatter(logging.Formatter):
    """
    Formats log records as structured JSON objects.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, config.date_format),
            "level": record.levelname,
            "name": record.name,
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
            "thread": threading.current_thread().name,
            "process": os.getpid(),
            "message": record.getMessage()
        }
        
        # Handle exceptions
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Handle extra contextual data
        for key, value in getattr(record, "context", {}).items():
            log_data[key] = value
            
        # Ensure we use proper JSON escaping for values
        return json.dumps(log_data)

# =============================================================================
# Performance Tracking
# =============================================================================
class PerformanceTracker:
    """
    Tracks execution time of functions or code blocks.
    Supports both synchronous and asynchronous code.
    """
    def __init__(self):
        self.timers = {}
        self.async_timers = {}
        self.stats = {
            "calls": {},
            "total_time": {},
            "max_time": {},
            "min_time": {}
        }
        
    def start(self, name: str) -> None:
        """Start a named timer."""
        self.timers[name] = time.time()
        
    def stop(self, name: str) -> float:
        """
        Stop a named timer and return elapsed time.
        
        Args:
            name (str): The timer name to stop.
            
        Returns:
            float: Elapsed time in seconds.
        """
        if name not in self.timers:
            logging.warning(f"Timer '{name}' was never started.")
            return 0.0
            
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        # Update stats
        if name not in self.stats["calls"]:
            self.stats["calls"][name] = 0
            self.stats["total_time"][name] = 0.0
            self.stats["max_time"][name] = 0.0
            self.stats["min_time"][name] = float('inf')
            
        self.stats["calls"][name] += 1
        self.stats["total_time"][name] += elapsed
        self.stats["max_time"][name] = max(self.stats["max_time"][name], elapsed)
        self.stats["min_time"][name] = min(self.stats["min_time"][name], elapsed)
        
        return elapsed

    def log_elapsed(self, name: str, level: int = logging.DEBUG) -> float:
        """
        Log the elapsed time for a named timer at the specified level.
        
        Args:
            name (str): The timer name.
            level (int): Logging level to use.
            
        Returns:
            float: Elapsed time in seconds.
        """
        elapsed = self.stop(name)
        logging.log(level, f"Performance: {name} completed in {elapsed:.4f} seconds")
        return elapsed
        
    async def start_async(self, name: str) -> None:
        """Start an async timer."""
        self.async_timers[name] = time.time()
        
    async def stop_async(self, name: str) -> float:
        """Stop an async timer and return elapsed time."""
        if name not in self.async_timers:
            logging.warning(f"Async timer '{name}' was never started.")
            return 0.0
            
        elapsed = time.time() - self.async_timers[name]
        del self.async_timers[name]
        
        # Update stats (sharing the same stats with sync timers)
        if name not in self.stats["calls"]:
            self.stats["calls"][name] = 0
            self.stats["total_time"][name] = 0.0
            self.stats["max_time"][name] = 0.0
            self.stats["min_time"][name] = float('inf')
            
        self.stats["calls"][name] += 1
        self.stats["total_time"][name] += elapsed
        self.stats["max_time"][name] = max(self.stats["max_time"][name], elapsed)
        self.stats["min_time"][name] = min(self.stats["min_time"][name], elapsed)
        
        return elapsed
        
    async def log_elapsed_async(self, name: str, level: int = logging.DEBUG) -> float:
        """Log elapsed time for an async timer."""
        elapsed = await self.stop_async(name)
        logging.log(level, f"Performance (async): {name} completed in {elapsed:.4f} seconds")
        return elapsed
        
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for a timer or all timers."""
        if name:
            if name not in self.stats["calls"]:
                return {
                    "calls": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "max_time": 0.0,
                    "min_time": 0.0
                }
                
            calls = self.stats["calls"][name]
            total = self.stats["total_time"][name]
            avg = total / calls if calls > 0 else 0.0
            
            return {
                "calls": calls,
                "total_time": total,
                "avg_time": avg,
                "max_time": self.stats["max_time"][name],
                "min_time": self.stats["min_time"][name]
            }
        else:
            # Get stats for all timers
            result = {}
            for timer in self.stats["calls"].keys():
                result[timer] = self.get_stats(timer)
            return result
            
    def reset_stats(self, name: Optional[str] = None) -> None:
        """Reset statistics for a timer or all timers."""
        if name:
            if name in self.stats["calls"]:
                self.stats["calls"][name] = 0
                self.stats["total_time"][name] = 0.0
                self.stats["max_time"][name] = 0.0
                self.stats["min_time"][name] = float('inf')
        else:
            # Reset all stats
            self.stats = {
                "calls": {},
                "total_time": {},
                "max_time": {},
                "min_time": {}
            }

# Global performance tracker instance
performance = PerformanceTracker()

# =============================================================================
# Decorators for Performance Tracking
# =============================================================================
def log_performance(level: int = logging.DEBUG):
    """
    Decorator to track and log function execution time.
    Handles both sync and async functions.
    
    Args:
        level (int): Logging level to use.
        
    Returns:
        Callable: Decorator function.
    """
    def decorator(func):
        # Handle both async and sync functions
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not config.track_performance:
                    return await func(*args, **kwargs)
                    
                timer_name = f"{func.__module__}.{func.__name__}"
                await performance.start_async(timer_name)
                try:
                    return await func(*args, **kwargs)
                finally:
                    await performance.log_elapsed_async(timer_name, level)
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not config.track_performance:
                    return func(*args, **kwargs)
                    
                timer_name = f"{func.__module__}.{func.__name__}"
                performance.start(timer_name)
                try:
                    return func(*args, **kwargs)
                finally:
                    performance.log_elapsed(timer_name, level)
            return wrapper
    return decorator

# =============================================================================
# Context Managers for Performance Tracking
# =============================================================================
class LogPerformance:
    """
    Context manager for manual performance tracking.
    
    Usage:
        with LogPerformance("operation_name"):
            # code to measure
    """
    def __init__(self, name: str, level: int = logging.DEBUG):
        self.name = name
        self.level = level
        
    def __enter__(self):
        performance.start(self.name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        performance.log_elapsed(self.name, self.level)
        return False  # Don't suppress exceptions

class AsyncLogPerformance:
    """
    Async context manager for performance tracking in async code.
    
    Usage:
        async with AsyncLogPerformance("async_operation"):
            # async code to measure
    """
    def __init__(self, name: str, level: int = logging.DEBUG):
        self.name = name
        self.level = level
        
    async def __aenter__(self):
        await performance.start_async(self.name)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await performance.log_elapsed_async(self.name, self.level)
        return False  # Don't suppress exceptions

# =============================================================================
# Context Logger - Add contextual information to log records
# =============================================================================
class ContextLogger:
    """
    Enhances a logger with contextual information that will be added to all messages.
    
    Usage:
        logger = ContextLogger(logging.getLogger(__name__))
        logger.set_context(user="admin", action="login")
        logger.info("User logged in")  # Will include user and action context
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context = {}
        self.module_name = logger.name
        _MODULE_LOGGERS[logger.name] = logger
        
    def set_context(self, **context) -> None:
        """Set contextual data for all subsequent log messages."""
        self.context.update(context)
        
    def clear_context(self) -> None:
        """Clear all contextual data."""
        self.context.clear()
        
    def with_context(self, **context):
        """
        Create a new ContextLogger with additional context (doesn't modify original).
        
        Usage:
            request_logger = logger.with_context(request_id="123", user_id="456")
            request_logger.info("Request processed")
        """
        new_logger = ContextLogger(self.logger)
        new_logger.context = self.context.copy()
        new_logger.context.update(context)
        return new_logger
        
    def _log_with_context(self, level, msg, *args, **kwargs):
        extra = kwargs.get('extra', {})
        extra['context'] = self.context.copy()
        kwargs['extra'] = extra
        return self.logger.log(level, msg, *args, **kwargs)
        
    def debug(self, msg, *args, **kwargs):
        return self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        return self._log_with_context(logging.INFO, msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        return self._log_with_context(logging.WARNING, msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        return self._log_with_context(logging.ERROR, msg, *args, **kwargs)
        
    def critical(self, msg, *args, **kwargs):
        return self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
        
    def exception(self, msg, *args, **kwargs):
        return self._log_with_context(logging.ERROR, msg, *args, exc_info=True, **kwargs)
        
    def set_level(self, level: Union[int, str]) -> None:
        """Set the level for this logger."""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        self.logger.setLevel(level)
        # Also update the module level in the configuration
        config.module_levels[self.module_name] = level

# =============================================================================
# Main Configuration Function
# =============================================================================
def configure_logging(
    level: Optional[Union[int, str]] = None,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    enable_file_logging: Optional[bool] = None,
    enable_json_logging: Optional[bool] = None,
    format: Optional[str] = None,
    file_format: Optional[str] = None,
    json_format: Optional[str] = None,
    enable_colors: Optional[bool] = None,
    enable_emojis: Optional[bool] = None,
    module_levels: Optional[Dict[str, Union[int, str]]] = None,
    **kwargs
) -> None:
    """
    Configures Python logging to use a consistent format and level across modules.
    Supports colorized console output, file logging, and JSON structured logging.
    
    Args:
        level (Optional[Union[int, str]]): Root logger level (int or string like "DEBUG")
        log_file (Optional[str]): Name of the log file
        log_dir (Optional[str]): Directory for log files
        enable_file_logging (Optional[bool]): Whether to enable file logging
        enable_json_logging (Optional[bool]): Whether to enable JSON structured logging
        format (Optional[str]): Console log format string
        file_format (Optional[str]): File log format string
        json_format (Optional[str]): JSON log format string
        enable_colors (Optional[bool]): Whether to enable colored console output
        enable_emojis (Optional[bool]): Whether to enable emoji log level icons
        module_levels (Optional[Dict[str, Union[int, str]]]): Per-module log levels
        **kwargs: Additional configuration options
    """
    global _LOGGING_INITIALIZED
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
        
    # Update configuration with provided parameters
    updates = {}
    if level is not None:
        updates["level"] = level
    if log_file is not None:
        updates["log_file"] = log_file
    if log_dir is not None:
        updates["log_dir"] = log_dir
    if enable_file_logging is not None:
        updates["enable_file"] = enable_file_logging
    if enable_json_logging is not None:
        updates["enable_json"] = enable_json_logging
    if format is not None:
        updates["console_format"] = format
    if file_format is not None:
        updates["file_format"] = file_format
    if json_format is not None:
        updates["json_format"] = json_format
    if enable_colors is not None:
        updates["enable_colors"] = enable_colors
    if enable_emojis is not None:
        updates["enable_emojis"] = enable_emojis
        
    # Apply updates to the config object
    config.update(**updates)
    
    # Apply module-specific log levels
    if module_levels:
        for module, level in module_levels.items():
            config.set_module_level(module, level)
    
    # Create root logger and set level
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level)
    
    # Remove any existing handlers to avoid duplicates during reinitialization
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handlers = []
    
    # 1. Console Handler with colors and emojis if available
    if COLORLOG_AVAILABLE and config.enable_colors:
        # Create colorized console handler
        console_handler = colorlog.StreamHandler()
        color_formatter = colorlog.ColoredFormatter(
            add_emojis_to_format("%(log_color)s" + config.console_format),
            log_colors=TERMINAL_COLORS,
            datefmt=config.date_format
        )
        console_handler.setFormatter(color_formatter)
        handlers.append(console_handler)
    elif RICH_AVAILABLE:
        # Use Rich for advanced console formatting
        console = Console()
        rich_traceback_install(console=console)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            enable_link_path=True
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console_handler)
    else:
        # Standard console handler with emojis
        console_handler = logging.StreamHandler()
        emoji_formatter = EmojiLogFormatter(
            add_emojis_to_format(config.console_format),
            datefmt=config.date_format
        )
        console_handler.setFormatter(emoji_formatter)
        handlers.append(console_handler)
        
    # 2. File Handler (if enabled)
    if config.enable_file:
        log_dir = ensure_log_directory()
        log_file = os.path.join(log_dir, config.log_file)
        
        # Create a rotating file handler
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=config.max_size,
                backupCount=config.backup_count
            )
        except Exception as e:
            # Fall back to standard file handler if rotating handler fails
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not create RotatingFileHandler: {e}. Falling back to FileHandler.")
            file_handler = logging.FileHandler(log_file)
            
        file_formatter = logging.Formatter(config.file_format, datefmt=config.date_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
    # 3. JSON Structured Logging (if enabled)
    if config.enable_json:
        log_dir = ensure_log_directory()
        json_log_file = os.path.join(log_dir, config.json_log_file)
        
        try:
            from logging.handlers import RotatingFileHandler
            json_handler = RotatingFileHandler(
                json_log_file,
                maxBytes=config.max_size,
                backupCount=config.backup_count
            )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not create JSON RotatingFileHandler: {e}. Falling back to FileHandler.")
            json_handler = logging.FileHandler(json_log_file)
            
        json_formatter = StructuredLogFormatter()
        json_handler.setFormatter(json_formatter)
        handlers.append(json_handler)
    
    # Add all handlers to the root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Log platform information at startup
    platform_info = get_platform_info()
    logging.info(f"Logging initialized on {platform_info['system']} {platform_info['release']} "
                f"with Python {platform_info['python']}")

# =============================================================================
# Helper Functions to Get Enhanced Loggers
# =============================================================================
def get_logger(name: str) -> logging.Logger:
    """
    Get a standard logger with the configured settings.
    
    Args:
        name (str): Logger name, typically __name__.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)

def get_context_logger(name: str) -> ContextLogger:
    """
    Get an enhanced context logger that allows adding additional context to log messages.
    
    Args:
        name (str): Logger name, typically __name__.
        
    Returns:
        ContextLogger: Context-aware logger wrapper.
    """
    return ContextLogger(logging.getLogger(name))

# =============================================================================
# Automatic Configuration When Module is Imported
# =============================================================================
# This is optional, allowing automatic configuration when the module is imported
# Uncomment to enable this behavior
patch_record_factory()
configure_logging()

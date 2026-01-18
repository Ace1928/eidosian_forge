"""
Logging configuration system for Word Forge.

This module defines the configuration schema for the Word Forge logging system,
including log levels, formats, rotation policies, and output destinations.

Architecture:
    ┌─────────────────────┐
    │   LoggingConfig     │
    └───────────┬─────────┘
                │
    ┌───────────┴─────────┐
    │     Components      │
    └─────────────────────┘
    ┌─────┬─────┬─────┬───────┬─────┐
    │Level│Form │Dest │Rotation│Path │
    └─────┴─────┴─────┴───────┴─────┘
"""

from dataclasses import dataclass, replace
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, TypedDict

from word_forge.configs.config_essentials import (
    LOGS_ROOT,
    LogDestination,
    LogFormatTemplate,
    LoggingConfigError,
    LogLevel,
    LogRotationStrategy,
)
from word_forge.configs.config_types import EnvMapping


class RotationConfigDict(TypedDict):
    """
    Type definition for rotation configuration settings.

    Used as the return type for get_rotation_config() to provide
    type-safe access to rotation settings.

    Attributes:
        enabled: Whether log rotation is enabled
        strategy: The rotation strategy name (if enabled)
        max_size_mb: Maximum file size before rotation (if enabled)
        max_files: Maximum number of log files to keep (if enabled)
    """

    enabled: bool
    strategy: Optional[str]
    max_size_mb: Optional[int]
    max_files: Optional[int]


class PythonLoggingHandlerDict(TypedDict):
    """
    Type definition for Python logging handler configuration.

    Represents the structure of a handler configuration entry
    in the Python logging configuration dictionary.

    Attributes:
        class: The handler class name
        level: The logging level for this handler
        formatter: The formatter name for this handler
        stream: Stream to use (for StreamHandler)
        filename: Log file path (for file handlers)
        maxBytes: Maximum file size (for RotatingFileHandler)
        backupCount: Maximum backup file count
        when: Rotation time specification (for TimedRotatingFileHandler)
    """

    class_: str  # Using class_ as class is a reserved keyword
    level: str
    formatter: str
    stream: Optional[str]
    filename: Optional[str]
    maxBytes: Optional[int]
    backupCount: Optional[int]
    when: Optional[str]


class PythonLoggingConfigDict(TypedDict):
    """
    Type definition for Python logging configuration dictionary.

    Represents the full structure of a configuration dictionary
    compatible with logging.config.dictConfig().

    Attributes:
        version: The logging configuration format version
        disable_existing_loggers: Whether to disable existing loggers
        formatters: Dictionary of formatter configurations
        handlers: Dictionary of handler configurations
        loggers: Dictionary of logger configurations
    """

    version: int
    disable_existing_loggers: bool
    formatters: Dict[str, Dict[str, str]]
    handlers: Dict[str, Any]  # Using Any for handlers due to varying structure
    loggers: Dict[str, Dict[str, Any]]


@dataclass
class LoggingConfig:
    """
    Configuration for Word Forge logging system.

    Controls log levels, formats, file paths, and rotation strategies
    for the application's logging infrastructure.

    Attributes:
        level: Logging level threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format string
        file_path: Path to log file (None = console logging only)
        destination: Where logs should be sent (console, file, both, syslog)
        rotation_strategy: Log rotation method (size, time, none)
        max_file_size_mb: Maximum log file size before rotation in MB
        max_files: Maximum number of rotated log files to keep
        include_timestamp_in_filename: Whether to add timestamps to log filenames
        propagate_to_root: Whether to propagate logs to the root logger
        log_exceptions: Whether to automatically log uncaught exceptions
        log_initialization: Whether to log when configuration is initialized

    Usage:
        ```python
        from word_forge.config import config

        # Get log level
        level = config.logging.level

        # Get formatted log path
        log_path = config.logging.get_log_path_with_timestamp()

        # Check if file logging is enabled
        uses_file = config.logging.uses_file_logging

        # Create configuration for debugging
        debug_config = config.logging.with_level("DEBUG")
        ```
    """

    # Log level and format
    level: LogLevel = "INFO"
    format: str = LogFormatTemplate.STANDARD.value

    # Log file settings
    file_path: Optional[str] = str(LOGS_ROOT / "word_forge.log")
    destination: LogDestination = LogDestination.BOTH

    # Rotation settings
    rotation_strategy: LogRotationStrategy = LogRotationStrategy.SIZE
    max_file_size_mb: int = 10
    max_files: int = 5

    # Advanced options
    include_timestamp_in_filename: bool = False
    propagate_to_root: bool = False
    log_exceptions: bool = True
    log_initialization: bool = True

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_LOG_LEVEL": ("level", str),
        "WORD_FORGE_LOG_FILE": ("file_path", str),
        "WORD_FORGE_LOG_FORMAT": ("format", str),
        "WORD_FORGE_LOG_DESTINATION": ("destination", LogDestination),
        "WORD_FORGE_LOG_ROTATION": ("rotation_strategy", LogRotationStrategy),
        "WORD_FORGE_LOG_MAX_SIZE": ("max_file_size_mb", int),
        "WORD_FORGE_LOG_EXCEPTIONS": ("log_exceptions", bool),
    }

    # ==========================================
    # Cached Properties
    # ==========================================

    @cached_property
    def get_log_path(self) -> Optional[Path]:
        """
        Get log file path as Path object if set.

        Returns:
            Path: Path object representing the log file location,
                  or None if file logging is disabled

        Example:
            ```python
            config = LoggingConfig()
            path = config.get_log_path
            if path:
                print(f"Logs will be written to: {path}")
            else:
                print("File logging is disabled")
            ```
        """
        if not self.file_path:
            return None
        return Path(self.file_path)

    @cached_property
    def uses_file_logging(self) -> bool:
        """
        Determine if file logging is enabled based on configuration.

        Returns:
            bool: True if logs are written to a file, False otherwise

        Example:
            ```python
            config = LoggingConfig()
            if config.uses_file_logging:
                print(f"File logging enabled at: {config.get_log_path}")
            ```
        """
        return (
            self.destination in (LogDestination.FILE, LogDestination.BOTH)
            and self.file_path is not None
        )

    @cached_property
    def uses_console_logging(self) -> bool:
        """
        Determine if console logging is enabled based on configuration.

        Returns:
            bool: True if logs are written to console, False otherwise

        Example:
            ```python
            config = LoggingConfig()
            if config.uses_console_logging:
                print("Console logging is enabled")
            ```
        """
        return self.destination in (LogDestination.CONSOLE, LogDestination.BOTH)

    # ==========================================
    # Public Methods
    # ==========================================

    def get_log_path_with_timestamp(self) -> Optional[Path]:
        """
        Get log path with timestamp if that option is enabled.

        Generates a filename with timestamp inserted before extension if
        include_timestamp_in_filename is True.

        Returns:
            Path: Path with timestamp added, or regular path if not enabled,
                  or None if file_path is None

        Example:
            ```python
            config = LoggingConfig(include_timestamp_in_filename=True)
            timestamped_path = config.get_log_path_with_timestamp()
            # Result: /path/to/word_forge_20230401_120530.log
            ```
        """
        if not self.file_path:
            return None

        path = Path(self.file_path)

        if not self.include_timestamp_in_filename:
            return path

        # Insert timestamp before extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = path.stem
        suffix = path.suffix

        return path.with_name(f"{stem}_{timestamp}{suffix}")

    def with_level(self, level: LogLevel) -> "LoggingConfig":
        """
        Create a new configuration with modified log level.

        Args:
            level: New log level value

        Returns:
            LoggingConfig: New configuration instance with updated level

        Example:
            ```python
            config = LoggingConfig()
            debug_config = config.with_level("DEBUG")
            print(f"New log level: {debug_config.level}")
            ```
        """
        return self._create_modified_config(level=level)

    def with_format_template(self, template: LogFormatTemplate) -> "LoggingConfig":
        """
        Create a new configuration using a predefined format template.

        Args:
            template: Log format template to use

        Returns:
            LoggingConfig: New configuration instance with updated format

        Example:
            ```python
            config = LoggingConfig()
            detailed_config = config.with_format_template(LogFormatTemplate.DETAILED)
            print(f"New format: {detailed_config.format}")
            ```
        """
        return self._create_modified_config(format=template.value)

    def with_destination(self, destination: LogDestination) -> "LoggingConfig":
        """
        Create a new configuration with modified log destination.

        Args:
            destination: New log destination value

        Returns:
            LoggingConfig: New configuration instance with updated destination

        Example:
            ```python
            config = LoggingConfig()
            file_only = config.with_destination(LogDestination.FILE)
            print(f"Uses console: {file_only.uses_console_logging}")
            print(f"Uses file: {file_only.uses_file_logging}")
            ```
        """
        config = self._create_modified_config(destination=destination)

        # Handle special case for FILE destination with no file path
        if (
            destination in (LogDestination.FILE, LogDestination.BOTH)
            and not config.file_path
        ):
            config = self._create_modified_config(
                destination=destination, file_path=str(LOGS_ROOT / "word_forge.log")
            )

        return config

    def get_rotation_config(self) -> RotationConfigDict:
        """
        Get rotation-specific configuration parameters.

        Returns:
            RotationConfigDict: Dictionary with rotation settings

        Example:
            ```python
            config = LoggingConfig()
            rotation = config.get_rotation_config()

            if rotation["enabled"]:
                print(f"Rotation strategy: {rotation['strategy']}")
                print(f"Max size: {rotation['max_size_mb']} MB")
            ```
        """
        if not self.uses_file_logging:
            return RotationConfigDict(
                enabled=False, strategy=None, max_size_mb=None, max_files=None
            )

        return RotationConfigDict(
            enabled=self.rotation_strategy != LogRotationStrategy.NONE,
            strategy=self.rotation_strategy.value,
            max_size_mb=self.max_file_size_mb,
            max_files=self.max_files,
        )

    def validate(self) -> None:
        """
        Validate the configuration for consistency and correctness.

        Performs comprehensive validation of all settings, including:
        - Consistency between destination and file path
        - Positive values for size and count settings
        - Valid rotation settings

        Raises:
            LoggingConfigError: If any validation fails with detailed error message

        Example:
            ```python
            config = LoggingConfig(max_file_size_mb=-1)
            try:
                config.validate()
            except LoggingConfigError as e:
                print(f"Invalid configuration: {e}")
            ```
        """
        errors = []

        self._validate_destination_settings(errors)
        self._validate_size_settings(errors)
        self._validate_rotation_settings(errors)

        if errors:
            raise LoggingConfigError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

    def get_python_logging_config(self) -> Dict[str, Any]:
        """
        Convert configuration to Python's logging module configuration dict.

        Creates a configuration dictionary compatible with logging.config.dictConfig()
        based on the current settings.

        Returns:
            Dict[str, Any]: Configuration dictionary for Python's logging system

        Example:
            ```python
            import logging.config

            config = LoggingConfig()
            logging_dict = config.get_python_logging_config()
            logging.config.dictConfig(logging_dict)
            logger = logging.getLogger("word_forge")
            logger.info("Logging system initialized")
            ```
        """
        handlers = self._get_active_handlers()

        config: Dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"standard": {"format": self.format}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.level,
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "word_forge": {
                    "level": self.level,
                    "handlers": handlers,
                    "propagate": self.propagate_to_root,
                }
            },
        }

        if self.uses_file_logging and self.file_path:
            config["handlers"]["file"] = self._create_file_handler_config()

        return config

    # ==========================================
    # Private Helper Methods
    # ==========================================

    def _create_modified_config(self, **kwargs: Any) -> "LoggingConfig":
        """
        Create a new configuration with modified attributes.

        Args:
            **kwargs: Attribute name-value pairs to override

        Returns:
            LoggingConfig: New configuration instance with specified modifications
        """
        return replace(self, **kwargs)

    def _get_active_handlers(self) -> List[str]:
        """
        Get list of active handler names based on configuration.

        Returns:
            List[str]: List of active handler names
        """
        handlers = []

        if self.uses_console_logging:
            handlers.append("console")

        if self.uses_file_logging:
            handlers.append("file")

        return handlers

    def _create_file_handler_config(self) -> Dict[str, Any]:
        """
        Create appropriate file handler configuration based on rotation settings.

        Returns:
            Dict[str, Any]: Handler configuration dictionary
        """
        if self.rotation_strategy == LogRotationStrategy.SIZE:
            return {
                "class": "logging.handlers.RotatingFileHandler",
                "level": self.level,
                "formatter": "standard",
                "filename": self.file_path,
                "maxBytes": self.max_file_size_mb * 1024 * 1024,
                "backupCount": self.max_files,
            }
        elif self.rotation_strategy == LogRotationStrategy.TIME:
            return {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": self.level,
                "formatter": "standard",
                "filename": self.file_path,
                "when": "midnight",
                "backupCount": self.max_files,
            }
        else:
            return {
                "class": "logging.FileHandler",
                "level": self.level,
                "formatter": "standard",
                "filename": self.file_path,
            }

    def _validate_destination_settings(self, errors: List[str]) -> None:
        """
        Validate settings related to logging destination.

        Args:
            errors: List to accumulate validation errors
        """
        # Validate destination vs file_path consistency
        if (
            self.destination in (LogDestination.FILE, LogDestination.BOTH)
            and not self.file_path
        ):
            errors.append("File logging enabled but no file path specified")

    def _validate_size_settings(self, errors: List[str]) -> None:
        """
        Validate settings related to size limitations.

        Args:
            errors: List to accumulate validation errors
        """
        # Validate max file size
        if self.max_file_size_mb <= 0:
            errors.append(
                f"Maximum file size must be positive, got {self.max_file_size_mb}"
            )

        # Validate max files
        if self.max_files <= 0:
            errors.append(
                f"Maximum number of files must be positive, got {self.max_files}"
            )

    def _validate_rotation_settings(self, errors: List[str]) -> None:
        """
        Validate settings related to log rotation.

        Args:
            errors: List to accumulate validation errors
        """
        # Validate rotation settings
        if (
            self.rotation_strategy == LogRotationStrategy.SIZE
            and self.uses_file_logging
            and self.max_file_size_mb <= 0
        ):
            errors.append("Size-based rotation requires positive max_file_size_mb")


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    "LoggingConfig",
    "LogLevel",
    "LogFormatTemplate",
    "LogRotationStrategy",
    "LogDestination",
    "LoggingConfigError",
    "RotationConfigDict",
]

"""
Logging Configuration System for Word Forge
===========================================

This module implements a comprehensive, type-safe configuration system for Word Forge's
logging infrastructure. It provides a flexible, consistent interface for controlling
log output formats, destinations, levels, and rotation policies.

Key Features:
- Type-safe configuration with validation
- Environment variable support for runtime configuration
- Multiple output destinations (console, file, both)
- Log rotation with size and time-based strategies
- Structured logging format templates
- Performance metrics for logging operations
- Complete compatibility with Python's standard logging module


Classes:
    - LoggingConfig: Central configuration class with immutable settings
    - Various TypedDict classes for type-safe configuration structures

Usage Examples:
        # Basic configuration
        config = create_default_logging_config()

        # Environment-specific configurations

        # Functional modification pattern
        debug_config = config.with_level(logging.DEBUG)
        rotated_config = config.with_rotation(LogRotationStrategy.SIZE, max_size_mb=20)

        # Integration with Python's logging system

Design Notes:
    - Uses immutable configuration pattern with functional modifications
    - Provides comprehensive validation before configuration application
    - Supports metrics collection for performance monitoring
    - Implements proper error handling with Result types
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

import logging
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cached_property, wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    FrozenSet,
    List,
    Optional,
    Protocol,
    TypedDict,
    cast,
)

from word_forge.configs.config_essentials import (
    LOGS_ROOT,
    EnvMapping,
    ErrorCategory,
    ErrorSeverity,
    LogDestination,
    LogFormatTemplate,
    LoggingConfigError,
    LogLevel,
    LogRotationStrategy,
    R,
    Result,
    measure_execution,
)


class RotationConfigDict(TypedDict, total=True):
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


class PythonLoggingFormatterDict(TypedDict, total=True):
    """
    Type definition for Python logging formatter configuration.

    Attributes:
        format: Format string for log messages
    """

    format: str


class PythonLoggingHandlerDict(TypedDict, total=False):
    """
    Type definition for Python logging handler configuration.

    Represents the structure of a handler configuration entry
    in the Python logging configuration dictionary.

    Attributes:
        class: The handler class name (using 'class' directly as it's in a dict)
        level: The logging level for this handler
        formatter: The formatter name for this handler
        stream: Stream to use (for StreamHandler)
        filename: Log file path (for file handlers)
        maxBytes: Maximum file size (for RotatingFileHandler)
        backupCount: Maximum backup file count
        when: Rotation time specification (for TimedRotatingFileHandler)
    """

    level: str
    formatter: str
    stream: Optional[str]
    filename: Optional[str]
    maxBytes: Optional[int]
    backupCount: Optional[int]
    when: Optional[str]


class PythonLoggingLoggerDict(TypedDict, total=True):
    """
    Type definition for Python logging logger configuration.

    Attributes:
        level: Logging level for this logger
        handlers: List of handler names for this logger
        propagate: Whether to propagate logs to parent loggers
    """

    level: str
    handlers: List[str]
    propagate: bool


class PythonLoggingConfigDict(TypedDict, total=True):
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
    formatters: Dict[str, PythonLoggingFormatterDict]
    handlers: Dict[str, Dict[str, Any]]  # Using Any due to varying structure
    loggers: Dict[str, PythonLoggingLoggerDict]


class LoggingMetrics(TypedDict, total=False):
    """
    Metrics for the logging system operation.

    Attributes:
        log_creation_time_ms: Time to create log entry in milliseconds
        handler_processing_time_ms: Time for handlers to process log in milliseconds
        message_size_bytes: Size of log message in bytes
        formatter_processing_time_ms: Time to format log message in milliseconds
    """

    log_creation_time_ms: float
    handler_processing_time_ms: float
    message_size_bytes: int
    formatter_processing_time_ms: float


class ValidatorMethod(Protocol):
    """Protocol for validation methods within the LoggingConfig class."""

    def __call__(self, instance: "LoggingConfig", errors: List[str]) -> None: ...


# ==========================================
# Constants for Validation
# ==========================================

DEFAULT_MAX_FILE_SIZE_MB: Final[int] = 10
DEFAULT_MAX_FILES: Final[int] = 5
MIN_FILE_SIZE_MB: Final[int] = 1
MIN_FILES: Final[int] = 1

VALID_LOG_LEVELS: FrozenSet[int] = frozenset(
    [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
)
DEFAULT_LOGGER_NAME: Final[str] = "word_forge"

# Logging
LoggingConfigDict = Dict[str, Any]  # Type alias for logging configuration
ValidationError = str  # Type alias for validation error messages
FormatStr = str  # Type alias for log format strings
LogFilePathStr = Optional[str]  # Type alias for log file path

# Function type for validation handlers
ValidationFunction = Callable[["LoggingConfig", List[ValidationError]], None]

# ==========================================
# Helper Functions and Decorators
# ==========================================


def validate_not_empty(value: Optional[str], error_message: str) -> Result[str]:
    """
    Validate that a string value is not None or empty.

    Args:
        value: String value to validate
        error_message: Error message if validation fails

    Returns:
        Result containing validated string or error
    """
    if not value:
        return Result[str].failure(
            code="VALIDATION_ERROR",
            message=error_message,
            context={
                "value": str(value) if value is not None else "",
                "error_message": error_message,
            },
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
        )
    return Result[str].success(value)


def with_metrics(operation_name: str) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to measure performance of logging configuration methods.

    Args:
        operation_name: Name of the operation for metrics collection

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            with measure_execution(f"logging.config.{operation_name}") as metrics:
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                metrics.duration_ms = (time.perf_counter() - start_time) * 1000
                return result

        return wrapper

    return decorator


# ==========================================
# Main Configuration Class
# ==========================================


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
    level: LogLevel = logging.INFO
    format: str = LogFormatTemplate.STANDARD.value

    # Log file settings
    file_path: Optional[str] = str(LOGS_ROOT / "word_forge.log")
    destination: LogDestination = LogDestination.BOTH

    # Rotation settings
    rotation_strategy: LogRotationStrategy = LogRotationStrategy.SIZE
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    max_files: int = DEFAULT_MAX_FILES

    # Advanced options
    include_timestamp_in_filename: bool = False
    propagate_to_root: bool = False
    log_exceptions: bool = True
    log_initialization: bool = True

    # Internal tracking
    _validators: List[ValidatorMethod] = field(default_factory=list, repr=False)
    _last_validation_errors: List[str] = field(default_factory=list, repr=False)
    _metrics: Dict[str, LoggingMetrics] = field(default_factory=dict, repr=False)

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_LOG_LEVEL": ("level", str),
        "WORD_FORGE_LOG_FILE": ("file_path", str),
        "WORD_FORGE_LOG_FORMAT": ("format", str),
        "WORD_FORGE_LOG_DESTINATION": ("destination", LogDestination),
        "WORD_FORGE_LOG_ROTATION": ("rotation_strategy", LogRotationStrategy),
        "WORD_FORGE_LOG_MAX_SIZE": ("max_file_size_mb", int),
        "WORD_FORGE_LOG_MAX_FILES": ("max_files", int),
        "WORD_FORGE_LOG_EXCEPTIONS": ("log_exceptions", bool),
        "WORD_FORGE_LOG_INIT": ("log_initialization", bool),
    }

    def __post_init__(self) -> None:
        """
        Initialize validators and perform initial configuration setup.
        """
        # Register validators
        self._validators = [
            self._validate_destination_settings,
            self._validate_size_settings,
            self._validate_rotation_settings,
            self._validate_level_settings,
        ]

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

    @cached_property
    def effective_log_path(self) -> Optional[Path]:
        """
        Get the actual log path that will be used, applying all configuration settings.

        Returns:
            Path: The effective log file path, or None if file logging is disabled
        """
        if not self.uses_file_logging or not self.file_path:
            return None

        if self.include_timestamp_in_filename:
            return self.get_log_path_with_timestamp()

        return Path(self.file_path)

    # ==========================================
    # Public Methods
    # ==========================================

    @with_metrics("get_log_path_with_timestamp")
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

    def with_file_path(self, file_path: Optional[str]) -> "LoggingConfig":
        """
        Create a new configuration with a different log file path.

        Args:
            file_path: New file path, or None to disable file logging

        Returns:
            LoggingConfig: New configuration instance with updated file path

        Example:
            ```python
            config = LoggingConfig()
            new_config = config.with_file_path("/var/log/word_forge.log")
            ```
        """
        # If setting to None, ensure destination is updated accordingly
        if file_path is None:
            dest = LogDestination.CONSOLE
        else:
            # If we currently don't use files, but now we're adding a file
            if not self.uses_file_logging:
                dest = LogDestination.BOTH
            else:
                # Keep current destination
                dest = self.destination

        return self._create_modified_config(file_path=file_path, destination=dest)

    def with_rotation(
        self,
        strategy: LogRotationStrategy,
        max_size_mb: Optional[int] = None,
        max_files: Optional[int] = None,
    ) -> "LoggingConfig":
        """
        Create a new configuration with modified rotation settings.

        Args:
            strategy: Log rotation strategy
            max_size_mb: Maximum file size in MB (for SIZE rotation)
            max_files: Maximum number of backup files to keep

        Returns:
            LoggingConfig: New configuration with updated rotation settings

        Example:
            ```python
            config = LoggingConfig()
            rotated_config = config.with_rotation(
                LogRotationStrategy.SIZE,
                max_size_mb=20,
                max_files=10
            )
            ```
        """
        kwargs: Dict[str, Any] = {"rotation_strategy": strategy}

        if max_size_mb is not None:
            kwargs["max_file_size_mb"] = max_size_mb

        if max_files is not None:
            kwargs["max_files"] = max_files

        return self._create_modified_config(**kwargs)

    @with_metrics("get_rotation_config")
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

    @with_metrics("validate")
    def validate(self) -> Result[None]:
        """
        Validate the configuration for consistency and correctness.

        Performs comprehensive validation of all settings, including:
        - Consistency between destination and file path
        - Positive values for size and count settings
        - Valid rotation settings
        - Valid log level

        Returns:
            Result indicating success or containing detailed error information

        Example:
            ```python
            config = LoggingConfig(max_file_size_mb=-1)
            result = config.validate()
            if result.is_failure:
                print(f"Invalid configuration: {result.error.message}")
            ```
        """
        errors: List[str] = []

        # Run all registered validators
        for validator in self._validators:
            validator(self, errors)

        # Store validation errors for later reference
        self._last_validation_errors = errors.copy()

        if errors:
            error_message = "; ".join(errors)
            return Result[None].failure(
                code="VALIDATION_ERROR",
                message=error_message,
                context={
                    "errors": "; ".join(errors),
                },
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
            )

        return Result[None].success(None)

    @with_metrics("get_python_logging_config")
    def get_python_logging_config(self) -> PythonLoggingConfigDict:
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

        config = cast(
            PythonLoggingConfigDict,
            {
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
                    DEFAULT_LOGGER_NAME: {
                        "level": self.level,
                        "handlers": handlers,
                        "propagate": self.propagate_to_root,
                    }
                },
            },
        )

        if self.uses_file_logging and self.file_path:
            config["handlers"]["file"] = self._create_file_handler_config()

        return config

    def get_validation_errors(self) -> List[str]:
        """
        Get list of validation errors from the last validation run.

        Returns:
            List of validation error messages

        Example:
            ```python
            config = LoggingConfig(max_file_size_mb=-1)
            config.validate()
            errors = config.get_validation_errors()
            for error in errors:
                print(f"- {error}")
            ```
        """
        return self._last_validation_errors.copy()

    def get_metrics(self) -> Dict[str, LoggingMetrics]:
        """
        Get metrics collected during logging configuration operations.

        Returns:
            Dictionary of operation metrics

        Example:
            ```python
            config = LoggingConfig()
            config.validate()
            metrics = config.get_metrics()
            print(f"Validation time: {metrics.get('validate', {}).get('duration_ms', 0)} ms")
            ```
        """
        return self._metrics.copy()

    def create_directory_if_needed(self) -> Result[None]:
        """
        Create directory for log file if it doesn't exist.

        Returns:
            Result indicating success or containing error information

        Example:
            ```python
            config = LoggingConfig()
            result = config.create_directory_if_needed()
            if result.is_failure:
                print(f"Failed to create log directory: {result.error.message}")
            ```
        """
        if not self.uses_file_logging or not self.file_path:
            return Result[None].success(None)

        log_path = Path(self.file_path)
        log_dir = log_path.parent

        try:
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
            return Result[None].success(None)
        except Exception as e:
            return Result[None].failure(
                code="DIRECTORY_CREATION_ERROR",
                message=f"Failed to create log directory: {str(e)}",
                context={
                    "log_dir": str(log_dir),
                    "error": str(e),
                },
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
            )

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
        handlers: List[str] = []

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

    def _validate_destination_settings(
        self, instance: "LoggingConfig", errors: List[str]
    ) -> None:
        """
        Validate settings related to logging destination.

        Args:
            instance: The configuration instance being validated
            errors: List to accumulate validation errors
        """
        # Validate destination vs file_path consistency
        if (
            instance.destination in (LogDestination.FILE, LogDestination.BOTH)
            and not instance.file_path
        ):
            errors.append("File logging enabled but no file path specified")

        # Validate file path is potentially writable if specified
        if instance.file_path:
            try:
                path = Path(instance.file_path)
                parent_dir = path.parent

                # Check if directory exists or can be created
                if not parent_dir.exists() and not os.access(
                    os.path.dirname(parent_dir), os.W_OK
                ):
                    errors.append(
                        f"Parent directory for log file is not writable: {parent_dir}"
                    )
            except Exception as e:
                errors.append(f"Invalid log file path: {str(e)}")

    def _validate_size_settings(
        self, instance: "LoggingConfig", errors: List[str]
    ) -> None:
        """
        Validate settings related to size limitations.

        Args:
            instance: The configuration instance being validated
            errors: List to accumulate validation errors
        """
        # Validate max file size
        if instance.max_file_size_mb <= 0:
            errors.append(
                f"Maximum file size must be positive, got {instance.max_file_size_mb}"
            )
        elif instance.max_file_size_mb < MIN_FILE_SIZE_MB:
            errors.append(
                f"Maximum file size should be at least {MIN_FILE_SIZE_MB}MB, got {instance.max_file_size_mb}MB"
            )

        # Validate max files
        if instance.max_files <= 0:
            errors.append(
                f"Maximum number of files must be positive, got {instance.max_files}"
            )
        elif instance.max_files < MIN_FILES:
            errors.append(
                f"Maximum number of files should be at least {MIN_FILES}, got {instance.max_files}"
            )

    def _validate_rotation_settings(
        self, instance: "LoggingConfig", errors: List[str]
    ) -> None:
        """
        Validate settings related to log rotation.

        Args:
            instance: The configuration instance being validated
            errors: List to accumulate validation errors
        """
        # Validate rotation settings
        if (
            instance.rotation_strategy == LogRotationStrategy.SIZE
            and instance.uses_file_logging
            and instance.max_file_size_mb <= 0
        ):
            errors.append("Size-based rotation requires positive max_file_size_mb")

        # Validate time-based rotation has reasonable backup count
        if (
            instance.rotation_strategy == LogRotationStrategy.TIME
            and instance.uses_file_logging
            and instance.max_files < 2
        ):
            errors.append("Time-based rotation should keep at least 2 backup files")

    def _validate_level_settings(
        self, instance: "LoggingConfig", errors: List[str]
    ) -> None:
        """
        Validate settings related to logging level.

        Args:
            instance: The configuration instance being validated
            errors: List to accumulate validation errors
        """
        # Ensure log level is valid
        if instance.level not in VALID_LOG_LEVELS:
            errors.append(
                f"Invalid log level: {instance.level}. Must be one of {', '.join(str(level) for level in VALID_LOG_LEVELS)}"
            )


# ==========================================
# Utility Functions
# ==========================================


def create_default_logging_config() -> LoggingConfig:
    """
    Create default logging configuration with standard settings.

    Returns:
        LoggingConfig: Default logging configuration instance

    Example:
        ```python
        default_config = create_default_logging_config()
        ```
    """
    return LoggingConfig()


def create_development_logging_config() -> LoggingConfig:
    """
    Create logging configuration optimized for development environments.

    Returns:
        LoggingConfig: Development-optimized logging configuration

    Example:
        ```python
        dev_config = create_development_logging_config()
        ```
    """
    return LoggingConfig(
        level=logging.DEBUG,
        format=LogFormatTemplate.DETAILED.value,
        destination=LogDestination.BOTH,
        rotation_strategy=LogRotationStrategy.SIZE,
        max_file_size_mb=5,
        max_files=3,
        log_initialization=True,
        log_exceptions=True,
    )


def create_production_logging_config() -> LoggingConfig:
    """
    Create logging configuration optimized for production environments.

    Returns:
        LoggingConfig: Production-optimized logging configuration

    Example:
        ```python
        prod_config = create_production_logging_config()
        ```
    """
    return LoggingConfig(
        level=logging.INFO,
        format=LogFormatTemplate.STANDARD.value,
        destination=LogDestination.BOTH,
        rotation_strategy=LogRotationStrategy.TIME,
        max_file_size_mb=20,
        max_files=14,  # Two weeks of logs
        log_initialization=True,
        log_exceptions=True,
        include_timestamp_in_filename=True,
    )


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Core configuration class
    "LoggingConfig",
    # Type definitions from config_essentials
    "LogLevel",
    "LogFormatTemplate",
    "LogRotationStrategy",
    "LogDestination",
    "LoggingConfigError",
    # Type definitions from this module
    "RotationConfigDict",
    "PythonLoggingConfigDict",
    # Factory functions
    "create_default_logging_config",
    "create_development_logging_config",
    "create_production_logging_config",
]

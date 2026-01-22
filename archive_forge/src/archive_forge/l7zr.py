# Import statements enhanced for clarity and completeness
import sys  # Provides access to system-specific parameters and functions. https://docs.python.org/3/library/sys.html
import os  # Provides a way of using operating system dependent functionality. https://docs.python.org/3/library/os.html
import datetime  # Supplies classes for manipulating dates and times. https://docs.python.org/3/library/datetime.html
import logging  # Defines functions and classes which implement a flexible event logging system. https://docs.python.org/3/library/logging.html
from typing import (  # Defines a standard notation for Python function and variable type annotations. https://docs.python.org/3/library/typing.html
    TextIO,  # A generic version of typing.IO[str]. https://docs.python.org/3/library/typing.html#typing.TextIO
    Optional,  # Optional type. https://docs.python.org/3/library/typing.html#typing.Optional
    Any,  # Special type indicating an unconstrained type. https://docs.python.org/3/library/typing.html#typing.Any
    Callable,  # Callable type; Callable[[int], str] is a function of (int) -> str. https://docs.python.org/3/library/typing.html#typing.Callable
    Type,  # A variable annotated with C may accept a value of type C. In contrast, a variable annotated with Type[C] may accept values that are classes themselves – specifically, it will accept the class object of C. https://docs.python.org/3/library/typing.html#typing.Type
    Tuple,  # Tuple type; Tuple[X, Y] is the type of a tuple of two items with the first item of type X and the second of type Y. https://docs.python.org/3/library/typing.html#typing.Tuple
    Dict,  # A generic version of dict. https://docs.python.org/3/library/typing.html#typing.Dict
    Union,  # Union type; Union[X, Y] means either X or Y. https://docs.python.org/3/library/typing.html#typing.Union
    Literal,  # Special typing form to define literal types. https://docs.python.org/3/library/typing.html#typing.Literal
    Iterable,  # A generic version of collections.abc.Iterable. https://docs.python.org/3/library/typing.html#typing.Iterable
    TypeVar,  # Type variable. https://docs.python.org/3/library/typing.html#typing.TypeVar
    Generic,  # Abstract base class for generic types. https://docs.python.org/3/library/typing.html#typing.Generic
    overload,  # Function decorator for defining overloaded functions. https://docs.python.org/3/library/typing.html#typing.overload
    List,  # A generic version of list. https://docs.python.org/3/library/typing.html#typing.List
    Protocol,  # Base class for protocol classes. https://docs.python.org/3/library/typing.html#typing.Protocol
    TypeAlias,  # Special annotation for explicitly declaring a type alias. https://docs.python.org/3/library/typing.html#typing.TypeAlias
)
import traceback  # Provides a standard interface to extract, format and print stack traces of Python programs. https://docs.python.org/3/library/traceback.html
import json  # Implements a subset of the JSON (JavaScript Object Notation) data interchange format. https://docs.python.org/3/library/json.html
from pathlib import (
    Path,
)  # Object-oriented filesystem paths. https://docs.python.org/3/library/pathlib.html
from collections.abc import (
    Iterable as IterableABC,
)  # Abstract base classes for containers. https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
from functools import (
    partial,
    singledispatch,
)  # Higher-order functions and operations on callable objects. https://docs.python.org/3/library/functools.html
from dataclasses import (
    dataclass,
    field,
    asdict,
)  # Provides a decorator and functions for automatically adding generated special methods to user-defined classes. https://docs.python.org/3/library/dataclasses.html
from enum import (
    Enum,
    auto,
    unique,
)  # Support for enumerations. https://docs.python.org/3/library/enum.html
from typing_extensions import (  # Provides additional facilities to the typing module. https://pypi.org/project/typing-extensions/
    Literal,  # Special typing form to define literal types. https://docs.python.org/3/library/typing.html#typing.Literal
    Self,  # Special type to represent the current class type. https://peps.python.org/pep-0673/
    TypeAlias,  # Special annotation for explicitly declaring a type alias. https://docs.python.org/3/library/typing.html#typing.TypeAlias
    Unpack,  # Special typing construct to unpack a variadic type. https://peps.python.org/pep-0646/
    ParamSpec,  # Parameter specification variable. https://peps.python.org/pep-0612/
    TypeVarTuple,  # Type variable tuple. https://peps.python.org/pep-0646/
    Protocol,  # Base class for protocol classes. https://docs.python.org/3/library/typing.html#typing.Protocol
)
from logging.handlers import QueueHandler, QueueListener
import queue
import asyncio
import multiprocessing
import concurrent.futures
from enum import Enum
from typing import TypeAlias, Optional, Tuple, Dict, Union, Literal
from logging import LogRecord


import logging
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

# Type Aliases
LogMessage = Union[str, Dict[str, Any]]
FormatterType = Union[logging.Formatter, None]

from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Literal, TypeAlias
from logging import LogRecord
# Type Aliases
LogMessage = Union[str, Dict[str, Any]]
FormatterType = Callable[[LogMessage], str]

import logging
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

# Type Aliases
LogMessage = Union[str, Dict[str, Any]]
FormatterType = Callable[[LogMessage], str]

import logging
from typing import Any, Dict, Optional, Union, Callable, TypeVar

# Define a type variable for the logger function
LoggerFunctionType = Callable[[str], None]

# Define a custom type for the configuration dictionary
ConfigDictType = Dict[str, Union[str, int, bool]]

# Centralises and Encapsulates all of the Aliasing and Configuration for the Logging System
class LoggerConfig:
    """
    Encapsulates configuration, enums, and type aliases for the logging system.
    This allows for easy collapsibility in IDEs and better code organization.
    """

    # CONSTANTS
    # Default Log File Directory
    LOG_DIR: Path = Path("/home/lloyd/EVIE/scripts/trading_bot/logs")
    # Default Log File Paths
    LOG_FILEPATHS: Dict[str, Path] = {
        "ASCII": LOG_DIR / "ASCII_log.txt",
        "coloured": LOG_DIR / "Coloured_ASCII_log.txt",
        "collapsible": LOG_DIR / "Collapsible_Coloured_ASCII_log.txt",
        "JSON": LOG_DIR / "Complex_JSON_log.json",
    }
    # Default Log File Names
    LOG_FILENAMES: Dict[str, str] = {
        "ASCII": "ASCII_log.txt",
        "coloured": "Coloured_ASCII_log.txt",
        "collapsible": "Collapsible_Coloured_ASCII_log.txt",
        "JSON": "Complex_JSON_log.json",
    }

    class LogLevel(Enum):
        NOTSET = "NOTSET"
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

    # Correcting and refining type aliases
    Filename: TypeAlias = str
    Mode: TypeAlias = Literal["r", "w", "a", "r+", "w+", "a+", "x"]
    Color: TypeAlias = Tuple[int, int, int]
    LogEntryMessageType: TypeAlias = str
    LogEntryLevelType: TypeAlias = LogLevel  # Directly using LogLevel enum
    LogEntryModuleType: TypeAlias = Optional[str]
    LogEntryFunctionType: TypeAlias = Optional[str]
    LogEntryLineType: TypeAlias = Optional[int]
    LogEntryExceptionType: TypeAlias = Optional[Tuple[str, List[str]]]
    LogEntryDictType: TypeAlias = Dict[
        str,
        Union[
            str,
            LogEntryLevelType,
            LogEntryModuleType,
            LogEntryFunctionType,
            LogEntryLineType,
            LogEntryExceptionType,
        ],
    ]
    LogEntryType: TypeAlias = LogRecord
    TimestampType: TypeAlias = str  # ISO 8601 format %Y-%m-%dT%H:%M:%S.%f

# Contains the methods for capturing enumeration information for logging to json, extensible to handle more types in the future.
class JsonLogTranscoder(json.JSONEncoder):
    """
    A custom JSON encoder that supports serialization of additional types.
    """

    def default(self, obj: Any) -> Any:
        """
        Serialize additional types to JSON.
        """
        # Custom serialization for Enum types
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        # Potential place for future custom serialization logic
        # Insert additional elif statements for other custom types here
        # Fallback to the base class implementation for other types
        return super().default(obj)

    @staticmethod
    def as_enum(dct: Dict[str, Any]) -> Any:
        """
        Deserialize JSON objects to Enum types.
        """
        if "__enum__" in dct:
            name, member = dct["__enum__"].split(".")
            return getattr(Enum, name)[member]
        return dct

# Contains the methods for capturing log information, default level debug, in a specified format for further procesing by the program. utilising the LoggingConfig class for all settings
class BasicLogger:
    """
    A basic logger class that captures log messages and splits them up into specific dictionary format including additional contextual metadata.
    """

    def __init__(
        self,
        timestamp: LoggerConfig.TimestampType = datetime.datetime.now().isoformat(),
        message: LoggerConfig.LogEntryMessageType = "",
        level: LoggerConfig.LogEntryLevelType = LoggerConfig.LogLevel.DEBUG,
        module: LoggerConfig.LogEntryModuleType = None,
        function: LoggerConfig.LogEntryFunctionType = None,
        line: LoggerConfig.LogEntryLineType = None,
        exc_info: LoggerConfig.LogEntryExceptionType = None,
    ) -> None:
        """
        Initializes a new instance of the BasicLogger class.
        Args:
            timestamp (TimestampType): The timestamp of the log entry. Defaults to the current timestamp.
            message (LogEntryMessageType): The log message to be captured.
            level (LogEntryLevelType): The log level of the message. Defaults to "DEBUG".
            module (LogEntryModuleType): The module where the log message originated. Defaults to None.
            function (LogEntryFunctionType): The function where the log message originated. Defaults to None.
            line (LogEntryLineType): The line number where the log message originated. Defaults to None.
            exc_info (LogEntryExceptionType): The exception information associated with the log message. Defaults to None.
        """
        self.timestamp: LoggerConfig.TimestampType = timestamp
        self.message: LoggerConfig.LogEntryMessageType = message
        self.level: LoggerConfig.LogEntryLevelType = level
        self.module: LoggerConfig.LogEntryModuleType = module
        self.function: LoggerConfig.LogEntryFunctionType = function
        self.line: LoggerConfig.LogEntryLineType = line
        self.exc_info: LoggerConfig.LogEntryExceptionType = exc_info

    def to_dict(self) -> LoggerConfig.LogEntryDictType:
        """
        Converts the LogEntry instance to a dictionary suitable for JSON serialization.
        Returns:
            JsonDict: The LogEntry instance as a dictionary.
        """
        return {
            "timestamp": self.timestamp,
            "message": self.message,
            "level": self.level,
            "module": self.module,
            "function": self.function,
            "line": self.line,
            "exc_info": self.exc_info,
        }

    def to_json(self) -> str:
        """
        Converts the LogEntry instance to a JSON string.
        Returns:
            str: The LogEntry instance as a JSON string.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)

# Takes the LogEntryDictType output from BasicLogger and returns a LogEntryDictType where each section of the LogEntryDict now contains fully formatted ASCII art graphics to enhance the formatting and presentation.
class AdvancedASCIIFormatter:
    """
    
    """
    def __init__(
        self,
        log: Optional[LoggerConfig.LogEntryDictType] = None,
    ) -> None:
        if log is None:
            log = {
                "timestamp": None,  # Placeholder for actual timestamp
                "message": None,  # Placeholder for actual message
                "level": None,  # Placeholder for actual level
                "module": None,  # Placeholder for actual module
                "function": None,  # Placeholder for actual function
                "line": None,  # Placeholder for actual line
                "exc_info": None,  # Placeholder for actual exception info
            }
        """
        Initializes the AdvancedASCIIFormatter class with the default ASCII art components.
        """
        self.log = log
        self.ENTRY_SEPARATOR: str = "-" * 80
        self.TOP_LEFT_CORNER: str = "┌"  # For Message Box Outline
        self.TOP_RIGHT_CORNER: str = "┐"  # For Message Box Outline
        self.BOTTOM_LEFT_CORNER: str = "└"  # For Message Box Outline
        self.BOTTOM_RIGHT_CORNER: str = "┘"  # For Message Box Outline
        self.HORIZONTAL_LINE: str = "─"  # For Message Box Outline
        self.VERTICAL_LINE: str = "│"  # For Message Box Outline
        self.HORIZONAL_DIVIDER_LEFT: str = (
            "├"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.HORIZONAL_DIVIDER_RIGHT: str = (
            "┤"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.HORIZONTAL_DIVIDER_MIDDLE: str = (
            "┼"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.VERTICAL_DIVIDER: str = (
            "┼"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_TOP_LEFT: str = (
            "╔"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_BOTTOM_LEFT: str = (
            "╚"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_TOP_RIGHT: str = (
            "╗"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_BOTTOM_RIGHT: str = (
            "╝"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_LEFT: str = (
            "╠"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_RIGHT: str = (
            "╣"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_MIDDLE: str = (
            "╬"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_VERTICAL: str = (
            "║"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_HORIZONTAL: str = (
            "═"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        # Level symbols for different log levels
        self.LEVEL_SYMBOLS: Dict[LoggerConfig.LogEntryLevelType, str] = {
            LoggerConfig.LogEntryLevelType.NOTSET: "🤷",
            LoggerConfig.LogEntryLevelType.DEBUG: "🐞",
            LoggerConfig.LogEntryLevelType.INFO: "ℹ️",
            LoggerConfig.LogEntryLevelType.WARNING: "⚠️",
            LoggerConfig.LogEntryLevelType.ERROR: "❌",
            LoggerConfig.LogEntryLevelType.CRITICAL: "🚨",
        }
        # Exception Symbols for different exception types
        self.EXCEPTION_SYMBOLS: Dict[Type[LoggerConfig.LogEntryExceptionType], str] = {
            ValueError: "#❌#",  # Value Error
            TypeError: "T❌T",  # Type Error
            KeyError: "K❌K",  # Key Error
            IndexError: "I❌I",  # Index Error
            AttributeError: "A❌A",  # Attribute Error
            Exception: "E❌E",  # General Exception
        }

    def __call__(self, log: LoggerConfig.LogEntryDictType) -> LoggerConfig.LogEntryDictType:
        """
        Prepares a log entry dictionary with advanced ASCII formatting for further processing.

        Args:
            log (LoggerConfig.LogEntryDictType): The log entry dictionary to process.

        Returns:
            LoggerConfig.LogEntryDictType: The processed log entry dictionary with enhanced formatting.
        """
        # Initialize a dictionary to hold the formatted log entry components
        formatted_log = {}

        # Format a header for marking clearly the start of the log entry
        formatted_log["header"] = self.ENTRY_SEPARATOR

        # Format the timestamp component of the log entry
        formatted_log["timestamp"] = f"{self.TOP_LEFT_CORNER} {log['timestamp']} {self.TOP_RIGHT_CORNER}"

        # Format the level component of the log entry
        formatted_log["level"] = f"{self.LEVEL_SYMBOLS[log['level']]} {log['level'].name.upper()}"

        # Format the message component of the log entry
        formatted_log["message"] = f"{self.VERTICAL_LINE} {log['message']}"

        # Format the module component of the log entry
        formatted_log["module"] = f"{self.HORIZONAL_DIVIDER_LEFT} Module: {log['module']} {self.HORIZONAL_DIVIDER_RIGHT}"

        # Format the function component of the log entry
        formatted_log["function"] = f"{self.HORIZONTAL_DIVIDER_MIDDLE} Function: {log['function']} {self.HORIZONTAL_DIVIDER_MIDDLE}"

        # Format the line component of the log entry
        formatted_log["line"] = f"{self.VERTICAL_LINE} Line: {log['line']}"

        # Format the exception component of the log entry
        formatted_log["exc_info"] = f"{self.ERROR_INDICATOR_BOX_TOP_LEFT} Exception: {log['exc_info']} {self.ERROR_INDICATOR_BOX_TOP_RIGHT}"

        # Add In the Authorship Details and a Footer Section to act as a clear separator and identifier for the log entry end
        formatted_log["authorship"] = f"{self.BOTTOM_LEFT_CORNER} Author: Lloyd Russell {self.BOTTOM_RIGHT_CORNER}"

        # Combine the exception and authorship components
        formatted_log["exc_info"] = [formatted_log["exc_info"] + "\n" + formatted_log["authorship"]]

        # Return the processed log entry dictionary with enhanced formatting
        return {
            "timestamp": formatted_log["timestamp"],
            "message": formatted_log["message"],
            "level": formatted_log["level"],
            "module": formatted_log["module"],
            "function": formatted_log["function"],
            "line": formatted_log["line"],
            "exc_info": formatted_log["exc_info"],
        }

    def format(self, log: LoggerConfig.LogEntryDictType) -> LoggerConfig.LogEntryDictType:
        """
        
        """
        # If the log entry is a string, process it as a message and split it into parts
        if isinstance(log, str):
            # Split the message into parts and process each part
            parts = self.split_message(log)
            # Initialize a dictionary to hold the formatted log entry components
            dict: LoggerConfig.LogEntryDictType = {}
            # Process each part of the message
            for part in parts:
                # Process the part and update the dictionary with the formatted components
                dict.update(self(part))
            # Return the dictionary containing the formatted log entry components
            return dict
        # If the log entry is a dictionary, process it as a log entry
        elif isinstance(log, dict):
            # Process the log entry and return the formatted dictionary
            return self(log)
        # If the log entry is neither a string nor a dictionary, raise a TypeError
        else:
            raise TypeError("Log entry must be a string or a dictionary")
        
class JsonFormatter:
    """
    
    Attributes:
        

    Methods:
        
        
    """

    def __call__(self, log: LoggerConfig.LogEntryDictType) -> LoggerConfig.LogEntryDictType:
        """
        
        """
        # Convert the log entry to a JSON string using the to_json() method
        json_log_entry: LoggerConfig.LogEntryDictType = log.to_json()

        # Return the JSON-formatted log entry
        return json_log_entry

class ColoredFormatter:
    """
    
    """
    def __init__(self, log: LoggerConfig.LogEntryDictType) -> None:
        """
        
        """
        self.COLORS: Dict[LoggerConfig.LogLevel, str] = {
            LoggerConfig.LogLevel.DEBUG: "\033[94m",
            LoggerConfig.LogLevel.INFO: "\033[92m",
            LoggerConfig.LogLevel.WARNING: "\033[93m",
            LoggerConfig.LogLevel.ERROR: "\033[91m",
            LoggerConfig.LogLevel.CRITICAL: "\033[95m",
        }
        self.RESET_COLOR: str = "\033[0m"
        self.log = log

    def __call__(
        self, log_entry: LoggerConfig.LogEntryDictType
    ) -> LoggerConfig.LogEntryDictType:
        """
        Formats a log entry with color based on its log level.

        Args:
            log_entry (Union[LogEntry, CollapsibleLogEntry]): The log entry to format.

        Returns:
            str: The formatted log entry with color.
        """
        color: str = self.COLORS.get(log.level, "")
        return super().__call__(
            f"{color}{log.timestamp} | {log.level.name.upper()} | {log.message}{self.RESET_COLOR}"
        )


class DualOutput(Generic[T]):
    """

    """

    def __init__(
        self,
        
    ) -> None:
        """

        """
        
    def write() 
        """
        """
        pass

    def _format_console_message()
        """
        
        """
        pass

    def _format_file_message()
        """
        
        """
        pass

class DualLogger:
    def __init__(self, name: str, config: Optional[ConfigDictType] = None) -> None:
        self.name: str = name
        self.config: ConfigDictType = config if config is not None else {}
        self.logger: logging.Logger = logging.getLogger(name)
        self._configure_logger()

    

    def log(self, level: Union[int, str], message: str) -> None:


class DualLogger:
    def __init__(self, name: str, file_level: int = logging.DEBUG, console_level: int = logging.DEBUG, formatter: FormatterType = None, config: Optional[Dict[str, Union[str, int, bool]]] = None) -> None:
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.file_handler: logging.FileHandler = logging.FileHandler(f"{name}.log")
        self.console_handler: logging.StreamHandler = logging.StreamHandler()
        self.file_handler.setLevel(file_level)
        self.console_handler.setLevel(console_level)
        self.formatter: logging.Formatter = formatter if formatter is not None else logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(self.formatter)
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)
        self.config: Dict[str, Union[str, int, bool]] = config if config is not None else {}
        self._configure_logger()

    def _configure_logger(self) -> None:
        log_level: Union[int, str] = self.config.get("log_level", logging.INFO)
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)

        log_format: str = self.config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        date_format: str = self.config.get("date_format", "%Y-%m-%d %H:%M:%S")
        formatter: logging.Formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

        stream_handler: logging.StreamHandler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        file_name: Optional[str] = self.config.get("file_name")
        if file_name:
            file_handler: logging.FileHandler = logging.FileHandler(file_name)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(self, level: int, message: LogMessage) -> None:
        if isinstance(message, dict):
            message = f"Details: {message}"
        self.logger.log(level, message)

    def notset(self, message: str) -> None:
        self.log(logging.NOTSET, message)

    def debug(self, message: str) -> None:
        self.log(logging.DEBUG, message)    

    def info(self, message: str) -> None:
        self.log(logging.INFO, message)

    def warning(self, message: str) -> None:
        self.log(logging.WARNING, message)

    def error(self, message: str) -> None:
        self.log(logging.ERROR, message)

    def critical(self, message: str) -> None:
        self.log(logging.CRITICAL, message)

    def set_file_level(self, level: int) -> None:
        self.file_handler.setLevel(level)

    def set_console_level(self, level: int) -> None:
        self.console_handler.setLevel(level)

    def add_custom_formatter(self, formatter: FormatterType) -> None:
        custom_formatter: logging.Formatter = logging.Formatter(formatter)
        self.file_handler.setFormatter(custom_formatter)
        self.console_handler.setFormatter(custom_formatter)

    


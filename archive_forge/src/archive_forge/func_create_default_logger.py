from __future__ import annotations
import re
import os
import sys
import logging
import typing
import traceback
import warnings
import pprint
import atexit as _atexit
import functools
import threading
from enum import Enum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Type, Union, Optional, Any, List, Dict, Tuple, Callable, Set, TYPE_CHECKING
def create_default_logger(name: Optional[str]=None, level: Union[str, int]='INFO', format: Optional[Callable]=None, filter: Optional[Callable]=None, handlers: Optional[List[logging.Handler]]=None, settings: Optional['BaseSettings']=None, **kwargs) -> Logger:
    """
    Creates a default logger
    """
    global _logger_contexts
    if name:
        if name.upper() in REVERSE_LOGLEVEL_MAPPING:
            level = name
            name = None
        else:
            name = extract_module_name(name)
    if name is None:
        name = 'lazyops'
    if name in _logger_contexts:
        return _logger_contexts[name]
    with _lock:
        if name == 'lazyops':
            return create_global_logger(name=name, level=level, format=format, filter=filter, handlers=handlers, settings=settings)
        if isinstance(level, str):
            level = level.upper()
        _logger = _logger_contexts['lazyops']
        if name and format is not None:

            def _filter_func(record: logging.LogRecord) -> bool:
                """
                Filter out messages from other modules
                """
                return extract_module_name(record.name) == name
            _logger.add(sys.stdout, enqueue=True, backtrace=True, colorize=True, level=level, format=format, filter=_filter_func, **kwargs)
            return _logger
        *options, extra = _logger._options
        new_logger = Logger(_logger._core, *options, {**extra})
        if name:
            _logger_contexts[name] = new_logger
            new_logger.name = name
            register_logger_module(name)
        if settings:
            new_logger.settings = settings
        return new_logger
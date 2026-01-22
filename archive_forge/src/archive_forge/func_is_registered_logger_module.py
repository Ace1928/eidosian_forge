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
@functools.lru_cache(maxsize=1000)
def is_registered_logger_module(name: str) -> bool:
    """
    Returns whether a logger module is registered
    """
    module_name = extract_module_name(name)
    return module_name in _registered_logger_modules
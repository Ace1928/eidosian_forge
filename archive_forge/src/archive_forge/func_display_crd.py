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
def display_crd(self, message: Any, *args, level: str='info', **kwargs):
    """
        Display CRD information in the log.

        Parameters
        ----------
        message : Any
            The message to display.
        level : str
            The log level to use.
        """
    __message = ''
    if isinstance(message, list):
        for m in message:
            if isinstance(m, dict):
                __message += ''.join((f'- <light-blue>{key}</>: {value}\n' for key, value in m.items()))
            else:
                __message += f'- <light-blue>{m}</>\n'
    elif isinstance(message, dict):
        __message = ''.join((f'- <light-blue>{key}</>: {value}\n' for key, value in message.items()))
    else:
        __message = str(message)
    _log = self.get_log_mode(level)
    _log(__message.strip(), *args, **kwargs)
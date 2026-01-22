import logging
import numbers
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from typing import AnyStr, Sequence  # noqa
from kombu.log import LOG_LEVELS
from kombu.log import get_logger as _get_logger
from kombu.utils.encoding import safe_str
from .term import colored
def get_task_logger(name):
    """Get logger for task module by name."""
    if name in RESERVED_LOGGER_NAMES:
        raise RuntimeError(f'Logger name {name!r} is reserved!')
    return _using_logger_parent(task_logger, get_logger(name))
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
def mlevel(level):
    """Convert level name/int to log level."""
    if level and (not isinstance(level, numbers.Integral)):
        return LOG_LEVELS[level.upper()]
    return level
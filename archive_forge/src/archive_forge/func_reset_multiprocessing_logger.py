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
def reset_multiprocessing_logger():
    """Reset multiprocessing logging setup."""
    try:
        from billiard import util
    except ImportError:
        pass
    else:
        if hasattr(util, '_logger'):
            util._logger = None
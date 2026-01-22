from __future__ import annotations
import argparse
import datetime
import functools
import logging
from typing import TYPE_CHECKING
def logged(level: int=logging.DEBUG) -> Callable:
    """
    Useful logging decorator. If a method is logged, the beginning and end of
    the method call will be logged at a pre-specified level.

    Args:
        level: Level to log method at. Defaults to DEBUG.
    """

    def wrap(f):
        _logger = logging.getLogger(f'{f.__module__}.{f.__name__}')

        def wrapped_f(*args, **kwargs):
            _logger.log(level, f'Called at {datetime.datetime.now()} with args = {args} and kwargs = {kwargs}')
            data = f(*args, **kwargs)
            _logger.log(level, f'Done at {datetime.datetime.now()} with args = {args} and kwargs = {kwargs}')
            return data
        return wrapped_f
    return wrap
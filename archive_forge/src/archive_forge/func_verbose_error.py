import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import click
import colorama
import ray  # noqa: F401
def verbose_error(self, msg: str, *args: Any, **kwargs: Any):
    """Logs an error if verbosity is not 0.

        For arguments, see `_format_msg`.
        """
    if self.verbosity > 0:
        self._error(msg, *args, _level_str='VERR', **kwargs)
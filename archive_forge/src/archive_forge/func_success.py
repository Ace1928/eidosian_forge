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
def success(self, msg: str, *args: Any, **kwargs: Any):
    """Prints a formatted success message.

        For arguments, see `_format_msg`.
        """
    self.print(cf.limeGreen(msg), *args, _level_str='SUCC', **kwargs)
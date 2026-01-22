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
def labeled_value(self, key: str, msg: str, *args: Any, **kwargs: Any):
    """Displays a key-value pair with special formatting.

        Args:
            key: Label that is prepended to the message.

        For other arguments, see `_format_msg`.
        """
    self._print(cf.skyBlue(key) + ': ' + _format_msg(cf.bold(msg), *args, **kwargs))
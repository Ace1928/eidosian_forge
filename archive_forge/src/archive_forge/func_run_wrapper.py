from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def run_wrapper(self, fn, *args, **kwargs):
    """Start the screen, call a function, then stop the screen.  Extra
        arguments are passed to `start`.

        Deprecated in favor of calling `start` as a context manager.
        """
    warnings.warn('run_wrapper is deprecated in favor of calling `start` as a context manager.', DeprecationWarning, stacklevel=3)
    with self.start(*args, **kwargs):
        return fn()
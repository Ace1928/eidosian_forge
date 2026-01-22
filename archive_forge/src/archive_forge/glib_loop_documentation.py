from __future__ import annotations
import functools
import logging
import signal
import typing
from gi.repository import GLib
from .abstract_loop import EventLoop, ExitMainLoop

        Decorator that cleanly exits the :class:`GLibEventLoop` if
        :exc:`ExitMainLoop` is thrown inside of the wrapped function. Store the
        exception info if some other exception occurs, it will be reraised after
        the loop quits.

        *f* -- function to be wrapped
        
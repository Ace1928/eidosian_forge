from __future__ import annotations
import contextlib
import functools
import logging
import selectors
import socket
import sys
import threading
import typing
from ctypes import byref
from ctypes.wintypes import DWORD
from urwid import signals
from . import _raw_display_base, _win32, escape
from .common import INPUT_DESCRIPTORS_CHANGED
def hook_event_loop(self, event_loop: EventLoop, callback: Callable[[list[str], list[int]], typing.Any]) -> None:
    """
        Register the given callback with the event loop, to be called with new
        input whenever it's available.  The callback should be passed a list of
        processed keys and a list of unprocessed keycodes.

        Subclasses may wish to use parse_input to wrap the callback.
        """
    self._input_thread = ReadInputThread(self._send_input, lambda: self._sigwinch_handler(28))
    self._input_thread.start()
    if hasattr(self, 'get_input_nonblocking'):
        wrapper = self._make_legacy_input_wrapper(event_loop, callback)
    else:

        @functools.wraps(callback)
        def wrapper() -> tuple[list[str], typing.Any] | None:
            return self.parse_input(event_loop, callback, self.get_available_raw_input())
    fds = self.get_input_descriptors()
    handles = [event_loop.watch_file(fd if isinstance(fd, int) else fd.fileno(), wrapper) for fd in fds]
    self._current_event_loop_handles = handles
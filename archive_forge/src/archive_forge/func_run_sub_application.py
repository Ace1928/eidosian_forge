from __future__ import unicode_literals
import functools
import os
import signal
import six
import sys
import textwrap
import threading
import time
import types
import weakref
from subprocess import Popen
from .application import Application, AbortAction
from .buffer import Buffer
from .buffer_mapping import BufferMapping
from .completion import CompleteEvent, get_common_complete_suffix
from .enums import SEARCH_BUFFER
from .eventloop.base import EventLoop
from .eventloop.callbacks import EventLoopCallbacks
from .filters import Condition
from .input import StdinInput, Input
from .key_binding.input_processor import InputProcessor
from .key_binding.input_processor import KeyPress
from .key_binding.registry import Registry
from .key_binding.vi_state import ViState
from .keys import Keys
from .output import Output
from .renderer import Renderer, print_tokens
from .search_state import SearchState
from .utils import Event
from .buffer import AcceptAction
def run_sub_application(self, application, done_callback=None, erase_when_done=False, _from_application_generator=False):
    """
        Run a sub :class:`~prompt_toolkit.application.Application`.

        This will suspend the main application and display the sub application
        until that one returns a value. The value is returned by calling
        `done_callback` with the result.

        The sub application will share the same I/O of the main application.
        That means, it uses the same input and output channels and it shares
        the same event loop.

        .. note:: Technically, it gets another Eventloop instance, but that is
            only a proxy to our main event loop. The reason is that calling
            'stop' --which returns the result of an application when it's
            done-- is handled differently.
        """
    assert isinstance(application, Application)
    assert done_callback is None or callable(done_callback)
    if self._sub_cli is not None:
        raise RuntimeError('Another sub application started already.')
    if not _from_application_generator:
        self.renderer.erase()

    def done():
        sub_cli._redraw()
        if erase_when_done or application.erase_when_done:
            sub_cli.renderer.erase()
        sub_cli.renderer.reset()
        sub_cli._is_running = False
        self._sub_cli = None
        if not _from_application_generator:
            self.renderer.request_absolute_cursor_position()
            self._redraw()
        if done_callback:
            done_callback(sub_cli.return_value())
    sub_cli = CommandLineInterface(application=application, eventloop=_SubApplicationEventLoop(self, done), input=self.input, output=self.output)
    sub_cli._is_running = True
    sub_cli._redraw()
    self._sub_cli = sub_cli
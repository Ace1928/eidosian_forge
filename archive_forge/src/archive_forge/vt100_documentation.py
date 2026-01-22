from __future__ import annotations
import sys
import contextlib
import io
import termios
import tty
from asyncio import AbstractEventLoop, get_running_loop
from typing import Callable, ContextManager, Generator, TextIO
from ..key_binding import KeyPress
from .base import Input
from .posix_utils import PosixStdinReader
from .vt100_parser import Vt100Parser
Wrapper around the callback that already removes the reader when
        the input is closed. Otherwise, we keep continuously calling this
        callback, until we leave the context manager (which can happen a bit
        later). This fixes issues when piping /dev/null into a prompt_toolkit
        application.
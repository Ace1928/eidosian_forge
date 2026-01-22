from __future__ import annotations
import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator, TextIO, cast
from .application import get_app_session, run_in_terminal
from .output import Output
def write_and_flush() -> None:
    self._output.enable_autowrap()
    if self.raw:
        self._output.write_raw(text)
    else:
        self._output.write(text)
    self._output.flush()
from __future__ import annotations
import atexit
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import weakref
from logging import Logger
from shutil import which as _which
from typing import Any
from tornado import gen
@gen.coroutine
def wait_async(self) -> Any:
    """Asynchronously wait for the process to finish."""
    proc = self.proc
    kill_event = self._kill_event
    while proc.poll() is None:
        if kill_event.is_set():
            self.terminate()
            msg = 'Process was aborted'
            raise ValueError(msg)
        yield gen.sleep(1.0)
    raise gen.Return(self.terminate())
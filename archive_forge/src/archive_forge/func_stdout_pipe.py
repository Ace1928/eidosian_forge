import contextlib
import fcntl
import itertools
import multiprocessing
import os
import pty
import re
import signal
import struct
import sys
import tempfile
import termios
import time
import traceback
import types
from typing import Optional, Generator, Tuple
import typing
def stdout_pipe(self) -> typing.TextIO:
    assert self._stdout_fifo_fd is not None
    stdout = os.fdopen(self._stdout_fifo_fd)
    self._stdout_fifo_fd = None
    return stdout
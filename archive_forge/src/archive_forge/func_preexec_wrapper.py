import os
import sys
import time
import pty
import tty
import errno
import signal
from contextlib import contextmanager
import ptyprocess
from ptyprocess.ptyprocess import use_native_pty_fork
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .spawnbase import SpawnBase
from .utils import (
def preexec_wrapper():
    """Set SIGHUP to be ignored, then call the real preexec_fn"""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if preexec_fn is not None:
        preexec_fn()
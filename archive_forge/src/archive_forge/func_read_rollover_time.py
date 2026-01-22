import datetime
import errno
import logging
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from io import TextIOWrapper
from logging.handlers import BaseRotatingHandler, TimedRotatingFileHandler
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Tuple
from portalocker import LOCK_EX, lock, unlock
import logging.handlers  # noqa: E402
def read_rollover_time(self) -> None:
    lock_file = self.clh.stream_lock
    if not lock_file or not self.clh.is_locked:
        self._console_log('No rollover time (lock) file to read from. Lock is not held?')
        return
    try:
        lock_file.seek(0)
        raw_time = lock_file.read()
    except OSError:
        self.rolloverAt = 0
        self._console_log(f"Couldn't read rollover time from file {lock_file!r}")
        return
    try:
        self.rolloverAt = int(raw_time.strip())
    except ValueError:
        self.rolloverAt = 0
        self._console_log(f"Couldn't read rollover time from file: {raw_time!r}")
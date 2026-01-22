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
def getLockFilename(self, lock_file_directory: Optional[str]) -> str:
    """
        Decide the lock filename. If the logfile is file.log, then we use `.__file.lock` and
        not `file.log.lock`. This only removes the extension if it's `*.log`.

        :param lock_file_directory: name of the directory for alternative living space of lock files
        :return: the path to the lock file.
        """
    lock_path, lock_name = self.baseLockFilename(self.baseFilename)
    if lock_file_directory:
        self.__create_lock_directory__(lock_file_directory)
        return os.path.join(lock_file_directory, lock_name)
    return os.path.join(lock_path, lock_name)
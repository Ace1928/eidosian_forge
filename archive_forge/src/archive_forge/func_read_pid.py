from __future__ import absolute_import
import errno
import os
import time
from . import (LockBase, AlreadyLocked, LockFailed, NotLocked, NotMyLock,
def read_pid(self):
    """ Get the PID from the lock file.
            """
    return read_pid_from_pidfile(self.path)
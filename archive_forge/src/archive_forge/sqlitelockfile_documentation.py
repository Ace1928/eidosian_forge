from __future__ import absolute_import, division
import time
import os
from . import LockBase, NotLocked, NotMyLock, LockTimeout, AlreadyLocked

        >>> lock = SQLiteLockFile('somefile')
        >>> lock = SQLiteLockFile('somefile', threaded=False)
        
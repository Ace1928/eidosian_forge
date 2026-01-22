from __future__ import absolute_import
import time
import os
from . import (LockBase, LockFailed, NotLocked, NotMyLock, LockTimeout,
Lock access to a file using atomic property of link(2).

    >>> lock = LinkLockFile('somefile')
    >>> lock = LinkLockFile('somefile', threaded=False)
    
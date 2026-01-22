from __future__ import absolute_import
import os
import time
from . import (LockBase, NotLocked, NotMyLock, LockTimeout,
Lock access to a file using symlink(2).
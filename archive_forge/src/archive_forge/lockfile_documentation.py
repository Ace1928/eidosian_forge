import errno
import os
from time import time as _uniquefloat
from twisted.python.runtime import platform
from os import rename

        Release this lock.

        This deletes the directory with the given name.

        @raise OSError: Any exception L{os.readlink()} may raise.
        @raise ValueError: If the lock is not owned by this process.
        
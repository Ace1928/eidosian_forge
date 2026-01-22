import errno
import os
from time import time as _uniquefloat
from twisted.python.runtime import platform
from os import rename
def isLocked(name):
    """
    Determine if the lock of the given name is held or not.

    @type name: C{str}
    @param name: The filesystem path to the lock to test

    @rtype: C{bool}
    @return: True if the lock is held, False otherwise.
    """
    l = FilesystemLock(name)
    result = None
    try:
        result = l.lock()
    finally:
        if result:
            l.unlock()
    return not result
import select
import sys
from errno import EBADF, EINTR
from time import sleep
from typing import Type
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
from twisted.python.runtime import platformType
def win32select(r, w, e, timeout=None):
    """Win32 select wrapper."""
    if not (r or w):
        if timeout is None:
            timeout = 0.01
        else:
            timeout = min(timeout, 0.001)
        sleep(timeout)
        return ([], [], [])
    if timeout is None or timeout > 0.5:
        timeout = 0.5
    r, w, e = select.select(r, w, w, timeout)
    return (r, w + e, [])
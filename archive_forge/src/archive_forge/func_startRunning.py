from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
def startRunning(self, installSignalHandlers: bool=True) -> None:
    """
        Start running the reactor, then kick off the timer that advances
        Twisted's clock to keep pace with CFRunLoop's.
        """
    super().startRunning(installSignalHandlers)
    self._scheduleSimulate(force=True)
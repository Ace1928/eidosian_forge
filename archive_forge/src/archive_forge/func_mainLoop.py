from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
def mainLoop(self) -> None:
    """
        Run the runner (C{CFRunLoopRun} or something that calls it), which runs
        the run loop until C{crash()} is called.
        """
    if not self._started:

        def docrash() -> None:
            self.crash()
        self._started = True
        self.callLater(0, docrash)
    already = False
    try:
        while self._started:
            if already:
                self._scheduleSimulate()
            already = True
            self._inCFLoop = True
            try:
                self._runner()
            finally:
                self._inCFLoop = False
    finally:
        self._stopSimulating()
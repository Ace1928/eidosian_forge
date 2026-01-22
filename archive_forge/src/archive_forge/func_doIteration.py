import socket
import sys
import warnings
from typing import Tuple, Type
from zope.interface import implementer
from twisted.internet import base, error, interfaces, main
from twisted.internet._dumbwin32proc import Process
from twisted.internet.iocpreactor import iocpsupport as _iocp, tcp, udp
from twisted.internet.iocpreactor.const import WAIT_TIMEOUT
from twisted.internet.win32eventreactor import _ThreadedWin32EventsMixin
from twisted.python import failure, log
def doIteration(self, timeout):
    """
        Poll the IO completion port for new events.
        """
    processed_events = 0
    if timeout is None:
        timeout = MAX_TIMEOUT
    else:
        timeout = min(MAX_TIMEOUT, int(1000 * timeout))
    rc, numBytes, key, evt = self.port.getEvent(timeout)
    while 1:
        if rc == WAIT_TIMEOUT:
            break
        if key != KEY_WAKEUP:
            assert key == KEY_NORMAL
            log.callWithLogger(evt.owner, self._callEventCallback, rc, numBytes, evt)
            processed_events += 1
        if processed_events >= EVENTS_PER_LOOP:
            break
        rc, numBytes, key, evt = self.port.getEvent(0)
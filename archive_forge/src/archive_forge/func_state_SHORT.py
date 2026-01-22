import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def state_SHORT(self, line):
    """
        Handle server responses for the SHORT state in which the server is
        expected to send a single line response.

        Parse the response and fire the deferred which is waiting on receipt of
        a complete response.  Transition the state back to WAITING.

        @type line: L{bytes}
        @param line: A line received from the server.

        @rtype: L{bytes}
        @return: The next state.
        """
    deferred, self._waiting = (self._waiting, None)
    self._unblock()
    code, status = _codeStatusSplit(line)
    if code == OK:
        deferred.callback(status)
    else:
        deferred.errback(ServerErrorResponse(status))
    return 'WAITING'
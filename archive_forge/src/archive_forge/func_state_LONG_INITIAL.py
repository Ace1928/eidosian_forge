import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def state_LONG_INITIAL(self, line):
    """
        Handle server responses for the LONG_INITIAL state in which the server
        is expected to send the first line of a multi-line response.

        Parse the response.  On an OK response, transition the state to
        LONG.  On an ERR response, cleanup and transition the state to
        WAITING.

        @type line: L{bytes}
        @param line: A line received from the server.

        @rtype: L{bytes}
        @return: The next state.
        """
    code, status = _codeStatusSplit(line)
    if code == OK:
        return 'LONG'
    consumer = self._consumer
    deferred = self._waiting
    self._consumer = self._waiting = self._xform = None
    self._unblock()
    deferred.errback(ServerErrorResponse(status, consumer))
    return 'WAITING'
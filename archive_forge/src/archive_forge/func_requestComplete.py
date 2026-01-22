import io
from collections import deque
from typing import List
from zope.interface import implementer
import h2.config
import h2.connection
import h2.errors
import h2.events
import h2.exceptions
import priority
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.policies import TimeoutMixin
from twisted.python.failure import Failure
from twisted.web.error import ExcessiveBufferingError
def requestComplete(self):
    """
        Called by the L{H2Connection} when the all data for a request has been
        received. Currently, with the legacy L{twisted.web.http.Request}
        object, just calls requestReceived unless the producer wants us to be
        quiet.
        """
    if self.producing:
        self._request.requestReceived(self.command, self.path, b'HTTP/2')
    else:
        self._inboundDataBuffer.append((_END_STREAM_SENTINEL, None))
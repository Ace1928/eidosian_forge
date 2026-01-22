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
def writeDataToStream(self, streamID, data):
    """
        May be called by L{H2Stream} objects to write response data to a given
        stream. Writes a single data frame.

        @param streamID: The ID of the stream to write the data to.
        @type streamID: L{int}

        @param data: The data chunk to write to the stream.
        @type data: L{bytes}
        """
    self._outboundStreamQueues[streamID].append(data)
    if self.conn.local_flow_control_window(streamID) > 0:
        self.priority.unblock(streamID)
        if self._sendingDeferred is not None:
            d = self._sendingDeferred
            self._sendingDeferred = None
            d.callback(streamID)
    if self.remainingOutboundWindow(streamID) <= 0:
        self.streams[streamID].flowControlBlocked()
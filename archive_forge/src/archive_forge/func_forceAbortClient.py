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
def forceAbortClient(self):
    """
        Called if C{abortTimeout} seconds have passed since the timeout fired,
        and the connection still hasn't gone away. This can really only happen
        on extremely bad connections or when clients are maliciously attempting
        to keep connections open.
        """
    self._log.info('Forcibly timing out client: {client}', client=self.transport.getPeer())
    self._abortingCall = None
    self.transport.abortConnection()
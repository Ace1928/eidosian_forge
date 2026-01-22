import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure
def checkState(self):
    """
        Check the state of the relay queue and, if possible, launch relayers to
        handle waiting messages.

        @rtype: L{None} or L{Deferred}
        @return: No return value if no further messages can be relayed or a
            deferred which fires when all of the SMTP connections initiated by
            this call have disconnected.
        """
    self.queue.readDirectory()
    if len(self.managed) >= self.maxConnections:
        return
    if not self.queue.hasWaiting():
        return
    return self._checkStateMX()
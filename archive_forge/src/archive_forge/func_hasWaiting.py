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
def hasWaiting(self):
    """
        Return an indication of whether the queue has messages waiting to be
        relayed.

        @rtype: L{bool}
        @return: C{True} if messages are waiting to be relayed.  C{False}
            otherwise.
        """
    return len(self.waiting) > 0
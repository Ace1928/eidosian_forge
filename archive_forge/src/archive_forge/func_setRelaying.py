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
def setRelaying(self, message):
    """
        Mark a message as being relayed.

        @type message: L{bytes}
        @param message: The base filename of a message.
        """
    del self.waiting[message]
    self.relayed[message] = 1
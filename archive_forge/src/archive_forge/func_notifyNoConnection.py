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
def notifyNoConnection(self, relay):
    """
        When a connection to the mail exchange server cannot be established,
        prepare to resend messages later.

        @type relay: L{SMTPManagedRelayerFactory}
        @param relay: The factory for the relayer meant to use the connection.
        """
    try:
        msgs = self.manager.managed[relay]
    except KeyError:
        log.msg('notifyNoConnection passed unknown relay!')
        return
    if self.noisy:
        log.msg('Backing off on delivery of ' + str(msgs))

    def setWaiting(queue, messages):
        map(queue.setWaiting, messages)
    self.reactor.callLater(30, setWaiting, self.manager.queue, msgs)
    del self.manager.managed[relay]
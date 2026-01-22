import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def setQueue(self, queue):
    """
        Set the queue for outgoing emails.

        @type queue: L{Queue}
        @param queue: A queue for outgoing messages.
        """
    self.queue = queue
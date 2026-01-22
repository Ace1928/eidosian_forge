import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def getPOP3Factory(self):
    """
        Create a POP3 protocol factory.

        @rtype: L{POP3Factory}
        @return: A POP3 protocol factory.
        """
    return protocols.POP3Factory(self)
import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def getESMTPFactory(self):
    """
        Create an ESMTP protocol factory.

        @rtype: L{ESMTPFactory <protocols.ESMTPFactory>}
        @return: An ESMTP protocol factory.
        """
    return protocols.ESMTPFactory(self, self.smtpPortal)
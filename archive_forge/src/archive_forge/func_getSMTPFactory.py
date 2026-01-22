import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def getSMTPFactory(self):
    """
        Create an SMTP protocol factory.

        @rtype: L{SMTPFactory <protocols.SMTPFactory>}
        @return: An SMTP protocol factory.
        """
    return protocols.SMTPFactory(self, self.smtpPortal)
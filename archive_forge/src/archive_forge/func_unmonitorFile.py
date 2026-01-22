import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def unmonitorFile(self, name):
    """
        Stop monitoring a file.

        @type name: L{bytes}
        @param name: A file name.
        """
    for i in range(len(self.files)):
        if name == self.files[i][1]:
            self.intervals.removeInterval(self.files[i][0])
            del self.files[i]
            break
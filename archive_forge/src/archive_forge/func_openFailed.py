from zope.interface import implementer
from twisted.internet import interfaces
from twisted.logger import Logger
from twisted.python import log
def openFailed(self, reason):
    """
        Called when the open failed for some reason.
        reason.desc is a string descrption, reason.code the SSH error code.

        @type reason: L{error.ConchError}
        """
    self._log.error('other side refused open\nreason: {reason}', reason=reason)
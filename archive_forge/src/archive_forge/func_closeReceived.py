from zope.interface import implementer
from twisted.internet import interfaces
from twisted.logger import Logger
from twisted.python import log
def closeReceived(self):
    """
        Called when the other side has closed the channel.
        """
    self._log.info('remote close')
    self.loseConnection()
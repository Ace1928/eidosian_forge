from zope.interface import implementer
from twisted.internet import interfaces
from twisted.logger import Logger
from twisted.python import log
def channelOpen(self, specificData):
    """
        Called when the channel is opened.  specificData is any data that the
        other side sent us when opening the channel.

        @type specificData: L{bytes}
        """
    self._log.info('channel open')
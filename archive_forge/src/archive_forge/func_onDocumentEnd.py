from sys import intern
from typing import Type
from twisted.internet import protocol
from twisted.python import failure
from twisted.words.xish import domish, utility
def onDocumentEnd(self):
    """Called whenever the end tag of the root element has been received.

        Closes the connection. This causes C{connectionLost} being called.
        """
    self.transport.loseConnection()
from sys import intern
from typing import Type
from twisted.internet import protocol
from twisted.python import failure
from twisted.words.xish import domish, utility
def removeBootstrap(self, event, fn):
    """
        Remove a bootstrap event handler.

        @param event: The event the observer is registered for.
        @type event: C{str} or L{xpath.XPathQuery}
        @param fn: The registered observer callable.
        """
    self.bootstraps.remove((event, fn))
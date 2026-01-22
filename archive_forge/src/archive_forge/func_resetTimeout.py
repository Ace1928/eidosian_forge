import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
def resetTimeout(self):
    """
        Reset the timeout count down.

        If the connection has already timed out, then do nothing.  If the
        timeout has been cancelled (probably using C{setTimeout(None)}), also
        do nothing.

        It's often a good idea to call this when the protocol has received
        some meaningful input from the other end of the connection.  "I've got
        some data, they're still there, reset the timeout".
        """
    if self.__timeoutCall is not None and self.timeOut is not None:
        self.__timeoutCall.reset(self.timeOut)
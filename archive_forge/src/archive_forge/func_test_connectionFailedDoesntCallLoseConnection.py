import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_connectionFailedDoesntCallLoseConnection(self):
    """
        L{ConnectedDatagramPort} does not call the deprecated C{loseConnection}
        in L{ConnectedDatagramPort.connectionFailed}.
        """

    def loseConnection():
        """
            Dummy C{loseConnection} method. C{loseConnection} is deprecated and
            should not get called.
            """
        self.fail('loseConnection is deprecated and should not get called.')
    port = unix.ConnectedDatagramPort(None, ClientProto())
    port.loseConnection = loseConnection
    port.connectionFailed('goodbye')
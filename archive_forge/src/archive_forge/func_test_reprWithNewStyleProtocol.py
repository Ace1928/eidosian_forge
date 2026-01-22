import os
import socket
import sys
from unittest import skipIf
from twisted.internet import address, defer, error, interfaces, protocol, reactor, utils
from twisted.python import lockfile
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.test.test_tcp import MyClientFactory, MyServerFactory
from twisted.trial import unittest
def test_reprWithNewStyleProtocol(self):
    """
        The two string representations of the L{IListeningPort} returned by
        L{IReactorUNIXDatagram.listenUNIXDatagram} contains the name of the
        new-style protocol class being used and the filename on which the port
        is listening or indicates that the port is not listening.
        """

    class NewStyleProtocol:

        def makeConnection(self, transport):
            pass

        def doStop(self):
            pass
    self.assertIsInstance(NewStyleProtocol, type)
    return self._reprTest(NewStyleProtocol(), 'twisted.test.test_unix.NewStyleProtocol')
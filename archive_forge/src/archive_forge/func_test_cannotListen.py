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
def test_cannotListen(self):
    """
        L{IReactorUNIXDatagram.listenUNIXDatagram} raises
        L{error.CannotListenError} if the unix socket specified is already in
        use.
        """
    addr = self.mktemp()
    p = ServerProto()
    s = reactor.listenUNIXDatagram(addr, p)
    self.assertRaises(error.CannotListenError, reactor.listenUNIXDatagram, addr, p)
    s.stopListening()
    os.unlink(addr)
from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
def test_socketsLeftOpen(self):
    f = protocol.Factory()
    f.protocol = protocol.Protocol
    reactor.listenTCP(0, f)
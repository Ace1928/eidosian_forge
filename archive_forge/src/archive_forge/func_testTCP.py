import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
@skipIf(not interfaces.IReactorTCP(reactor, None), 'IReactorTCP is needed')
def testTCP(self):
    p = reactor.listenTCP(0, protocol.ServerFactory())
    portNo = p.getHost().port
    self.assertNotEqual(str(p).find(str(portNo)), -1, '%d not found in %s' % (portNo, p))
    return p.stopListening()
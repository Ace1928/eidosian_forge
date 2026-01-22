import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def test_everythingThere(self):
    """
        L{twisted.application.internet} dynamically defines a set of
        L{service.Service} subclasses that in general have corresponding
        reactor.listenXXX or reactor.connectXXX calls.
        """
    trans = ['TCP', 'UNIX', 'SSL', 'UDP', 'UNIXDatagram', 'Multicast']
    for tran in trans[:]:
        if not getattr(interfaces, 'IReactor' + tran)(reactor, None):
            trans.remove(tran)
    for tran in trans:
        for side in ['Server', 'Client']:
            if tran == 'Multicast' and side == 'Client':
                continue
            if tran == 'UDP' and side == 'Client':
                continue
            self.assertTrue(hasattr(internet, tran + side))
            method = getattr(internet, tran + side).method
            prefix = {'Server': 'listen', 'Client': 'connect'}[side]
            self.assertTrue(hasattr(reactor, prefix + method) or (prefix == 'connect' and method == 'UDP'))
            o = getattr(internet, tran + side)()
            self.assertEqual(service.IService(o), o)
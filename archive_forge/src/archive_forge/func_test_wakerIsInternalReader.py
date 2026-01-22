import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_wakerIsInternalReader(self):
    """
        When L{PosixReactorBase} is instantiated, it creates a waker and adds
        it to its internal readers set.
        """
    reactor = TrivialReactor()
    self._checkWaker(reactor)
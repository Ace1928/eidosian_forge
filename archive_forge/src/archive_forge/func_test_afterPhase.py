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
def test_afterPhase(self):
    """
        L{IReactorCore.addSystemEventTrigger} should accept the C{'after'}
        phase and not call the given object until the right event is fired.
        """
    self._addSystemEventTriggerTest('after')
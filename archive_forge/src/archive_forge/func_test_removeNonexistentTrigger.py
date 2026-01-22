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
def test_removeNonexistentTrigger(self):
    """
        L{_ThreePhaseEvent.removeTrigger} should raise L{ValueError} when given
        an object not previously returned by L{_ThreePhaseEvent.addTrigger}.
        """
    self.assertRaises(ValueError, self.event.removeTrigger, object())
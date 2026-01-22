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
def test_addBeforeTrigger(self):
    """
        L{_ThreePhaseEvent.addTrigger} should accept C{'before'} as a phase, a
        callable, and some arguments and add the callable with the arguments to
        the before list.
        """
    self.event.addTrigger('before', self.trigger, self.arg)
    self.assertEqual(self.event.before, [(self.trigger, (self.arg,), {})])
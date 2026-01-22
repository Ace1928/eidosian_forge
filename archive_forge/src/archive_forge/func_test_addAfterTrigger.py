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
def test_addAfterTrigger(self):
    """
        L{_ThreePhaseEvent.addTrigger} should accept C{'after'} as a phase, a
        callable, and some arguments and add the callable with the arguments to
        the after list.
        """
    self.event.addTrigger('after', self.trigger, self.arg)
    self.assertEqual(self.event.after, [(self.trigger, (self.arg,), {})])
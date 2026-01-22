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
def test_removeAlmostValidTrigger(self):
    """
        L{_ThreePhaseEvent.removeTrigger} should raise L{ValueError} if it is
        given a trigger handle which resembles a valid trigger handle aside
        from its phase being incorrect.
        """
    self.assertRaises(KeyError, self.event.removeTrigger, ('xxx', self.trigger, (self.arg,), {}))
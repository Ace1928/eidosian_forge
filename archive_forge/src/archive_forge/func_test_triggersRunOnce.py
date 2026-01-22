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
def test_triggersRunOnce(self):
    """
        A trigger should only be called on the first call to
        L{_ThreePhaseEvent.fireEvent}.
        """
    events = []
    self.event.addTrigger('before', events.append, 'before')
    self.event.addTrigger('during', events.append, 'during')
    self.event.addTrigger('after', events.append, 'after')
    self.event.fireEvent()
    self.event.fireEvent()
    self.assertEqual(events, ['before', 'during', 'after'])
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
def test_synchronousBeforeRemovesDuring(self):
    """
        If a before-phase trigger removes a during-phase trigger, the
        during-phase trigger should not be run.
        """
    events = []
    self.event.addTrigger('before', lambda: self.event.removeTrigger(duringHandle))
    duringHandle = self.event.addTrigger('during', events.append, 'during')
    self.event.addTrigger('after', events.append, 'after')
    self.event.fireEvent()
    self.assertEqual(events, ['after'])
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
def test_synchronousRemoveAlreadyExecutedBefore(self):
    """
        If a before-phase trigger tries to remove another before-phase trigger
        which has already run, a warning should be emitted.
        """
    events = []

    def removeTrigger():
        self.event.removeTrigger(beforeHandle)
    beforeHandle = self.event.addTrigger('before', events.append, ('first', 'before'))
    self.event.addTrigger('before', removeTrigger)
    self.event.addTrigger('before', events.append, ('second', 'before'))
    self.assertWarns(DeprecationWarning, 'Removing already-fired system event triggers will raise an exception in a future version of Twisted.', __file__, self.event.fireEvent)
    self.assertEqual(events, [('first', 'before'), ('second', 'before')])
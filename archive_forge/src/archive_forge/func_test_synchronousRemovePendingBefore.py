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
def test_synchronousRemovePendingBefore(self):
    """
        If a before-phase trigger removes another before-phase trigger which
        has not yet run, the removed trigger should not be run.
        """
    events = []
    self.event.addTrigger('before', lambda: self.event.removeTrigger(beforeHandle))
    beforeHandle = self.event.addTrigger('before', events.append, ('first', 'before'))
    self.event.addTrigger('before', events.append, ('second', 'before'))
    self.event.fireEvent()
    self.assertEqual(events, [('second', 'before')])
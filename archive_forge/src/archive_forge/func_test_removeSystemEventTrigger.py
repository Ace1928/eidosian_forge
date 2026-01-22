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
def test_removeSystemEventTrigger(self):
    """
        A trigger removed with L{IReactorCore.removeSystemEventTrigger} should
        not be called when the event fires.
        """
    eventType = 'test'
    events = []

    def firstBeforeTrigger():
        events.append('first')

    def secondBeforeTrigger():
        events.append('second')
    self.addTrigger('before', eventType, firstBeforeTrigger)
    self.removeTrigger(self.addTrigger('before', eventType, secondBeforeTrigger))
    self.assertEqual(events, [])
    reactor.fireSystemEvent(eventType)
    self.assertEqual(events, ['first'])
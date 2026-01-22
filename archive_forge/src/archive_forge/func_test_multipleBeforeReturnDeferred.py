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
def test_multipleBeforeReturnDeferred(self):
    """
        If more than one trigger added to the C{'before'} phase of an event
        return L{Deferred}s, the C{'during'} phase should be delayed until they
        are all called back.
        """
    firstDeferred = Deferred()
    secondDeferred = Deferred()
    eventType = 'test'
    events = []

    def firstBeforeTrigger():
        return firstDeferred

    def secondBeforeTrigger():
        return secondDeferred

    def duringTrigger():
        events.append('during')
    self.addTrigger('before', eventType, firstBeforeTrigger)
    self.addTrigger('before', eventType, secondBeforeTrigger)
    self.addTrigger('during', eventType, duringTrigger)
    self.assertEqual(events, [])
    reactor.fireSystemEvent(eventType)
    self.assertEqual(events, [])
    firstDeferred.callback(None)
    self.assertEqual(events, [])
    secondDeferred.callback(None)
    self.assertEqual(events, ['during'])
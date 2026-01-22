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
def test_subsequentBeforeTriggerFiresPriorBeforeDeferred(self):
    """
        If a trigger added to the C{'before'} phase of an event calls back a
        L{Deferred} returned by an earlier trigger in the C{'before'} phase of
        the same event, the remaining C{'before'} triggers for that event
        should be run and any further L{Deferred}s waited on before proceeding
        to the C{'during'} events.
        """
    eventType = 'test'
    events = []
    firstDeferred = Deferred()
    secondDeferred = Deferred()

    def firstBeforeTrigger():
        return firstDeferred

    def secondBeforeTrigger():
        firstDeferred.callback(None)

    def thirdBeforeTrigger():
        events.append('before')
        return secondDeferred

    def duringTrigger():
        events.append('during')
    self.addTrigger('before', eventType, firstBeforeTrigger)
    self.addTrigger('before', eventType, secondBeforeTrigger)
    self.addTrigger('before', eventType, thirdBeforeTrigger)
    self.addTrigger('during', eventType, duringTrigger)
    self.assertEqual(events, [])
    reactor.fireSystemEvent(eventType)
    self.assertEqual(events, ['before'])
    secondDeferred.callback(None)
    self.assertEqual(events, ['before', 'during'])
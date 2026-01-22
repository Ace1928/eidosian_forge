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
def test_interactionBetweenDifferentEvents(self):
    """
        L{IReactorCore.fireSystemEvent} should behave the same way for a
        particular system event regardless of whether Deferreds are being
        waited on for a different system event.
        """
    events = []
    firstEvent = 'first-event'
    firstDeferred = Deferred()

    def beforeFirstEvent():
        events.append(('before', 'first'))
        return firstDeferred

    def afterFirstEvent():
        events.append(('after', 'first'))
    secondEvent = 'second-event'
    secondDeferred = Deferred()

    def beforeSecondEvent():
        events.append(('before', 'second'))
        return secondDeferred

    def afterSecondEvent():
        events.append(('after', 'second'))
    self.addTrigger('before', firstEvent, beforeFirstEvent)
    self.addTrigger('after', firstEvent, afterFirstEvent)
    self.addTrigger('before', secondEvent, beforeSecondEvent)
    self.addTrigger('after', secondEvent, afterSecondEvent)
    self.assertEqual(events, [])
    reactor.fireSystemEvent(firstEvent)
    self.assertEqual(events, [('before', 'first')])
    reactor.fireSystemEvent(secondEvent)
    self.assertEqual(events, [('before', 'first'), ('before', 'second')])
    firstDeferred.callback(None)
    self.assertEqual(events, [('before', 'first'), ('before', 'second'), ('after', 'first')])
    secondDeferred.callback(None)
    self.assertEqual(events, [('before', 'first'), ('before', 'second'), ('after', 'first'), ('after', 'second')])
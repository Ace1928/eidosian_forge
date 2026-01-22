from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testCallLaterDelayed(self):
    """
        Test that calls can be delayed.
        """
    events = []
    c = task.Clock()
    call = c.callLater(1, lambda a, b: events.append((a, b)), 1, b=2)
    call.delay(1)
    self.assertEqual(call.getTime(), 2)
    c.advance(1.5)
    self.assertEqual(events, [])
    c.advance(1.0)
    self.assertEqual(events, [(1, 2)])
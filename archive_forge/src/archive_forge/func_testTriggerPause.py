from twisted.protocols import pcp
from twisted.trial import unittest
def testTriggerPause(self):
    """Make sure I say "when." """
    self.proxy.pauseProducing()
    self.assertFalse(self.parentProducer.paused, "don't pause yet")
    self.proxy.write('x' * 51)
    self.assertFalse(self.parentProducer.paused, "don't pause yet")
    self.proxy.write('x' * 51)
    self.assertTrue(self.parentProducer.paused)
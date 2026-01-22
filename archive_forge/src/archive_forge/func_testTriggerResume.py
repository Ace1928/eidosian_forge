from twisted.protocols import pcp
from twisted.trial import unittest
def testTriggerResume(self):
    """Make sure I resumeProducing when my buffer empties."""
    self.proxy.pauseProducing()
    self.proxy.write('x' * 102)
    self.assertTrue(self.parentProducer.paused, 'should be paused')
    self.proxy.resumeProducing()
    self.assertFalse(self.parentProducer.paused, 'Producer should have resumed.')
    self.assertFalse(self.proxy.producerPaused)
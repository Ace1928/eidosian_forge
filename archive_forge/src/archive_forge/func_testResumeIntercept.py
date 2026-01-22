from twisted.protocols import pcp
from twisted.trial import unittest
def testResumeIntercept(self):
    self.proxy.pauseProducing()
    self.proxy.resumeProducing()
    self.assertFalse(self.parentProducer.resumed)
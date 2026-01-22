from twisted.protocols import pcp
from twisted.trial import unittest
def testLateWriteBuffering(self):
    self.proxy.resumeProducing()
    self.proxy.write('datum' * 21)
    self.assertEqual(self.underlying.getvalue(), 'datum' * 20)
    self.assertEqual(self.proxy._buffer, ['datum'])
from twisted.protocols import pcp
from twisted.trial import unittest
def testPull(self):
    self.proxy.write('hello')
    self.proxy.resumeProducing()
    self.assertEqual(self.underlying.getvalue(), 'hello')
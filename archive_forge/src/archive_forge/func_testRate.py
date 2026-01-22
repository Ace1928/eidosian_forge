from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
def testRate(self) -> None:
    delta_t = 10
    self.bucket.add(100)
    self.shaped.write('x' * 100)
    self.clock.set(delta_t)
    self.shaped.resumeProducing()
    self.assertEqual(len(self.underlying.getvalue()), delta_t * self.bucket.rate)
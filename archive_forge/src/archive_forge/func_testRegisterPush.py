from twisted.protocols import pcp
from twisted.trial import unittest
def testRegisterPush(self):
    self.consumer.registerProducer(self.producer, True)
    self.assertFalse(self.producer.resumed)
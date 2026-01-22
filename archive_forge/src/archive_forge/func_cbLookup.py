import time
from zope.interface.verify import verifyClass
from twisted.internet import interfaces, task
from twisted.names import cache, dns
from twisted.trial import unittest
def cbLookup(result):
    self.assertEqual(result[0][0].ttl, 59)
    self.assertEqual(result[1][0].ttl, 49)
    self.assertEqual(result[2][0].ttl, 39)
    self.assertEqual(result[0][0].name.name, b'example.com')
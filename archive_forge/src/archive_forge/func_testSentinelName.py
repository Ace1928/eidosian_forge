import unittest2 as unittest
from mock import sentinel, DEFAULT
def testSentinelName(self):
    self.assertEqual(str(sentinel.whatever), 'sentinel.whatever', 'sentinel name incorrect')
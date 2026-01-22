import unittest2 as unittest
from mock import sentinel, DEFAULT
def testSentinels(self):
    self.assertEqual(sentinel.whatever, sentinel.whatever, 'sentinel not stored')
    self.assertNotEqual(sentinel.whatever, sentinel.whateverelse, 'sentinel should be unique')
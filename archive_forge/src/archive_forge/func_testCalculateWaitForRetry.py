import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def testCalculateWaitForRetry(self):
    try0 = util.CalculateWaitForRetry(0)
    self.assertTrue(try0 >= 1.0)
    self.assertTrue(try0 <= 1.5)
    try1 = util.CalculateWaitForRetry(1)
    self.assertTrue(try1 >= 1.0)
    self.assertTrue(try1 <= 3.0)
    try2 = util.CalculateWaitForRetry(2)
    self.assertTrue(try2 >= 2.0)
    self.assertTrue(try2 <= 6.0)
    try3 = util.CalculateWaitForRetry(3)
    self.assertTrue(try3 >= 4.0)
    self.assertTrue(try3 <= 12.0)
    try4 = util.CalculateWaitForRetry(4)
    self.assertTrue(try4 >= 8.0)
    self.assertTrue(try4 <= 24.0)
    self.assertAlmostEqual(10, util.CalculateWaitForRetry(5, max_wait=10))
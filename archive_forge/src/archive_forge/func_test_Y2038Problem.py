import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_Y2038Problem(self):
    """
        L{SerialNumber} raises ArithmeticError when used to add dates more than
        68 years in the future.
        """
    maxAddTime = calendar.timegm(datetime(2038, 1, 19, 3, 14, 7).utctimetuple())
    self.assertEqual(maxAddTime, SerialNumber(0)._maxAdd)
    self.assertRaises(ArithmeticError, lambda: SerialNumber(0) + SerialNumber(maxAddTime + 1))
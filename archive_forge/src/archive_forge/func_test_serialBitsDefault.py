import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_serialBitsDefault(self):
    """
        L{SerialNumber.serialBits} has default value 32.
        """
    self.assertEqual(SerialNumber(1)._serialBits, 32)
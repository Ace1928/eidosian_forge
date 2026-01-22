import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_Y2106Problem(self):
    """
        L{SerialNumber} wraps unix timestamps in the year 2106.
        """
    self.assertEqual(SerialNumber(-1).toRFC4034DateString(), '21060207062815')
import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_unixEpoch(self):
    """
        L{SerialNumber.toRFC4034DateString} stores 32bit timestamps relative to
        the UNIX epoch.
        """
    self.assertEqual(SerialNumber(0).toRFC4034DateString(), '19700101000000')
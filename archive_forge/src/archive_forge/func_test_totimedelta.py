import unittest
import operator
from datetime import timedelta, date, datetime
from isodate import Duration, parse_duration, ISO8601Error
from isodate import D_DEFAULT, D_WEEK, D_ALT_EXT, duration_isoformat
def test_totimedelta(self):
    """
        Test conversion form Duration to timedelta.
        """
    dur = Duration(years=1, months=2, days=10)
    self.assertEqual(dur.totimedelta(datetime(1998, 2, 25)), timedelta(434))
    self.assertEqual(dur.totimedelta(datetime(2000, 2, 25)), timedelta(435))
    dur = Duration(months=2)
    self.assertEqual(dur.totimedelta(datetime(2000, 2, 25)), timedelta(60))
    self.assertEqual(dur.totimedelta(datetime(2001, 2, 25)), timedelta(59))
    self.assertEqual(dur.totimedelta(datetime(2001, 3, 25)), timedelta(61))
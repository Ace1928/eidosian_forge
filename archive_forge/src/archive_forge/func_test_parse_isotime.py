import datetime
import unittest
import pytz
from wsme import utils
def test_parse_isotime(self):
    good_times = [('12:03:54', datetime.time(12, 3, 54)), ('23:59:59.000004', datetime.time(23, 59, 59, 4)), ('01:02:03+00:00', datetime.time(1, 2, 3, 0, pytz.UTC)), ('01:02:03+23:59', datetime.time(1, 2, 3, 0, pytz.FixedOffset(1439))), ('01:02:03-23:59', datetime.time(1, 2, 3, 0, pytz.FixedOffset(-1439)))]
    ill_formatted_times = ['24-12-2004']
    out_of_range_times = ['32:12:00', '00:54:60', '01:02:03-24:00', '01:02:03+24:00']
    for s, t in good_times:
        assert utils.parse_isotime(s) == t
    for s in ill_formatted_times + out_of_range_times:
        self.assertRaises(ValueError, utils.parse_isotime, s)
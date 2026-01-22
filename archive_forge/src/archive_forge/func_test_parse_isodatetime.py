import datetime
import unittest
import pytz
from wsme import utils
def test_parse_isodatetime(self):
    good_datetimes = [('2008-02-12T12:03:54', datetime.datetime(2008, 2, 12, 12, 3, 54)), ('2012-05-14T23:59:59.000004', datetime.datetime(2012, 5, 14, 23, 59, 59, 4)), ('1856-07-10T01:02:03+00:00', datetime.datetime(1856, 7, 10, 1, 2, 3, 0, pytz.UTC)), ('1856-07-10T01:02:03+23:59', datetime.datetime(1856, 7, 10, 1, 2, 3, 0, pytz.FixedOffset(1439))), ('1856-07-10T01:02:03-23:59', datetime.datetime(1856, 7, 10, 1, 2, 3, 0, pytz.FixedOffset(-1439)))]
    ill_formatted_datetimes = ['24-32-2004', '1856-07-10+33:00']
    out_of_range_datetimes = ['2008-02-12T32:12:00', '2012-13-12T00:54:60']
    for s, t in good_datetimes:
        assert utils.parse_isodatetime(s) == t
    for s in ill_formatted_datetimes + out_of_range_datetimes:
        self.assertRaises(ValueError, utils.parse_isodatetime, s)
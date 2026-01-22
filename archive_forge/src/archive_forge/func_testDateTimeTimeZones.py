import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testDateTimeTimeZones(self):
    """Test that a datetime string with a timezone is decoded correctly."""
    tests = (('2012-09-30T15:31:50.262-06:00', (2012, 9, 30, 15, 31, 50, 262000, util.TimeZoneOffset(-360))), ('2012-09-30T15:31:50.262+01:30', (2012, 9, 30, 15, 31, 50, 262000, util.TimeZoneOffset(90))), ('2012-09-30T15:31:50+00:05', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(5))), ('2012-09-30T15:31:50+00:00', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(0))), ('2012-09-30t15:31:50-00:00', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(0))), ('2012-09-30t15:31:50z', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(0))), ('2012-09-30T15:31:50-23:00', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(-1380))))
    for datetime_string, datetime_vals in tests:
        decoded = util.decode_datetime(datetime_string)
        expected = datetime.datetime(*datetime_vals)
        self.assertEquals(expected, decoded)
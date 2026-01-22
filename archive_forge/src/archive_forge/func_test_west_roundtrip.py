import calendar
import datetime
from unittest import mock
import iso8601
from glance.common import timeutils
from glance.tests import utils as test_utils
def test_west_roundtrip(self):
    time_str = '2012-02-14T20:53:07+11:30'
    west = timeutils.parse_isotime(time_str)
    self.assertEqual(west.tzinfo.tzname(None), '+11:30')
    self.assertEqual(timeutils.isotime(west), time_str)
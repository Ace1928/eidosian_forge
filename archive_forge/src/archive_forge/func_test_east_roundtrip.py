import calendar
import datetime
from unittest import mock
import iso8601
from glance.common import timeutils
from glance.tests import utils as test_utils
def test_east_roundtrip(self):
    time_str = '2012-02-14T20:53:07-07:00'
    east = timeutils.parse_isotime(time_str)
    self.assertEqual(east.tzinfo.tzname(None), '-07:00')
    self.assertEqual(timeutils.isotime(east), time_str)
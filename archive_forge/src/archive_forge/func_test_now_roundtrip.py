import calendar
import datetime
from unittest import mock
import iso8601
from glance.common import timeutils
from glance.tests import utils as test_utils
def test_now_roundtrip(self):
    time_str = timeutils.isotime()
    now = timeutils.parse_isotime(time_str)
    self.assertEqual(now.tzinfo, iso8601.iso8601.UTC)
    self.assertEqual(timeutils.isotime(now), time_str)
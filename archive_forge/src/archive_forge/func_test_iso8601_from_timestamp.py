import calendar
import datetime
from unittest import mock
import iso8601
from glance.common import timeutils
from glance.tests import utils as test_utils
def test_iso8601_from_timestamp(self):
    utcnow = timeutils.utcnow()
    iso = timeutils.isotime(utcnow)
    ts = calendar.timegm(utcnow.timetuple())
    self.assertEqual(iso, timeutils.iso8601_from_timestamp(ts))
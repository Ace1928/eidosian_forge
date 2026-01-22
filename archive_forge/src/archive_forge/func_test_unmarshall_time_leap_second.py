import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_unmarshall_time_leap_second(self):
    leap_dict = dict(day=30, month=6, year=2015, hour=23, minute=59, second=timeutils._MAX_DATETIME_SEC + 1, microsecond=0)
    leap_time = timeutils.unmarshall_time(leap_dict)
    leap_dict.update(second=timeutils._MAX_DATETIME_SEC)
    expected = timeutils.unmarshall_time(leap_dict)
    self.assertEqual(expected, leap_time)
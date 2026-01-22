import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_marshall_time_with_tz(self):
    now = timeutils.utcnow()
    now = now.replace(tzinfo=iso8601.iso8601.UTC)
    binary = timeutils.marshall_now(now)
    self.assertEqual('UTC', binary['tzname'])
    backagain = timeutils.unmarshall_time(binary)
    self.assertEqual(now, backagain)
    self.assertIsNotNone(backagain.tzinfo)
    self.assertEqual(now.utcoffset(), backagain.utcoffset())
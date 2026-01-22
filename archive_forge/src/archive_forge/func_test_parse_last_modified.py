import datetime
from unittest import mock
from testtools import matchers
from heat.engine.clients.os import swift
from heat.tests import common
from heat.tests import utils
def test_parse_last_modified(self):
    self.assertIsNone(self.swift_plugin.parse_last_modified(None))
    if zoneinfo:
        tz = zoneinfo.ZoneInfo('GMT')
    else:
        tz = pytz.timezone('GMT')
    now = datetime.datetime(2015, 2, 5, 1, 4, 40, 0, tz)
    now_naive = datetime.datetime(2015, 2, 5, 1, 4, 40, 0)
    last_modified = now.strftime('%a, %d %b %Y %H:%M:%S %Z')
    self.assertEqual('Thu, 05 Feb 2015 01:04:40 GMT', last_modified)
    self.assertEqual(now_naive, self.swift_plugin.parse_last_modified(last_modified))
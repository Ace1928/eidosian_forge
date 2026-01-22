import unittest
from datetime import datetime
from .dateutil import UTC, TimezoneInfo, format_rfc3339, parse_rfc3339
def test_format_rfc3339(self):
    self.assertEqual(format_rfc3339(datetime(2017, 7, 25, 4, 44, 21, 0, UTC)), '2017-07-25T04:44:21Z')
    self.assertEqual(format_rfc3339(datetime(2017, 7, 25, 4, 44, 21, 0, TimezoneInfo(2, 0))), '2017-07-25T02:44:21Z')
    self.assertEqual(format_rfc3339(datetime(2017, 7, 25, 4, 44, 21, 0, TimezoneInfo(-2, 30))), '2017-07-25T07:14:21Z')
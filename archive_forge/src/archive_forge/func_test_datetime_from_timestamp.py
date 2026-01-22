import datetime
import time
from dateutil import tz
from testtools import matchers
import yaql.tests
def test_datetime_from_timestamp(self):
    dt = DT(2006, 11, 21, 16, 30, tzinfo=tz.tzutc())
    self.assertEqual(dt, self.eval('datetime(1164126600)'))
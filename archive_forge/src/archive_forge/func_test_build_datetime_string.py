import datetime
import time
from dateutil import tz
from testtools import matchers
import yaql.tests
def test_build_datetime_string(self):
    self.assertEqual(DT(2006, 11, 21, 16, 30, tzinfo=tz.tzutc()), self.eval('datetime("Tuesday, 21. November 2006 04:30PM", "%A, %d. %B %Y %I:%M%p")'))
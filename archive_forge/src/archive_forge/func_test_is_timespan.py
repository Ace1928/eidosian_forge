import datetime
import time
from dateutil import tz
from testtools import matchers
import yaql.tests
def test_is_timespan(self):
    self.assertTrue(self.eval('isTimespan(timespan(milliseconds => -1))'))
    self.assertFalse(self.eval('isTimespan(123)'))
    self.assertFalse(self.eval('isTimespan(abc)'))
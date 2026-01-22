import datetime
import time
from dateutil import tz
from testtools import matchers
import yaql.tests
def test_build_timespan(self):
    self.assertEqual(TS(0), self.eval('timespan()'))
    self.assertEqual(TS(1, 7384, 5006), self.eval('timespan(days => 1, hours => 2, minutes => 3, seconds => 4, milliseconds => 5, microseconds => 6)'))
    self.assertEqual(TS(1, 7384, 4994), self.eval('timespan(days => 1, hours => 2, minutes => 3, seconds =>4, milliseconds => 5, microseconds => -6)'))
    self.assertEqual(TS(microseconds=-1000), self.eval('timespan(milliseconds => -1)'))
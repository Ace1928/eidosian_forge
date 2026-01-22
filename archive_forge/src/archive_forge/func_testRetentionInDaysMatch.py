from __future__ import absolute_import
import gslib.tests.testcase as testcase
from gslib.utils.retention_util import _RetentionPeriodToString
from gslib.utils.retention_util import DaysToSeconds
from gslib.utils.retention_util import MonthsToSeconds
from gslib.utils.retention_util import RetentionInDaysMatch
from gslib.utils.retention_util import RetentionInMonthsMatch
from gslib.utils.retention_util import RetentionInSeconds
from gslib.utils.retention_util import RetentionInSecondsMatch
from gslib.utils.retention_util import RetentionInYearsMatch
from gslib.utils.retention_util import SECONDS_IN_DAY
from gslib.utils.retention_util import SECONDS_IN_MONTH
from gslib.utils.retention_util import SECONDS_IN_YEAR
from gslib.utils.retention_util import YearsToSeconds
def testRetentionInDaysMatch(self):
    days = '30d'
    days_match = RetentionInDaysMatch(days)
    self.assertEqual('30', days_match.group('number'))
    days = '1d'
    days_match = RetentionInDaysMatch(days)
    self.assertEqual('1', days_match.group('number'))
    days = '1day'
    days_match = RetentionInDaysMatch(days)
    self.assertEqual(None, days_match)
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
def testRetentionInMonthsMatch(self):
    months = '30m'
    months_match = RetentionInMonthsMatch(months)
    self.assertEqual('30', months_match.group('number'))
    months = '1m'
    months_match = RetentionInMonthsMatch(months)
    self.assertEqual('1', months_match.group('number'))
    months = '1month'
    months_match = RetentionInMonthsMatch(months)
    self.assertEqual(None, months_match)
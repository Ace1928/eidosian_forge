from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_user_returns_result(self):
    case = self.make_case()

    def returns():
        return 1
    run = RunTest(case)
    run.result = ExtendedTestResult()
    self.assertEqual(1, run._run_user(returns))
    self.assertEqual([], run.result._events)
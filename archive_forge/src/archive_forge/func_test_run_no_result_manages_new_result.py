from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test_run_no_result_manages_new_result(self):
    log = []
    run = RunTest(self.make_case(), lambda x: log.append(x) or x)
    result = run.run()
    self.assertIsInstance(result.decorated, TestResult)
from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test_run_with_result(self):
    log = []

    class Case(TestCase):

        def _run_test_method(self, result):
            log.append(result)
    case = Case('_run_test_method')
    run = RunTest(case, lambda x: log.append(x))
    result = TestResult()
    run.run(result)
    self.assertEqual(1, len(log))
    self.assertEqual(result, log[0].decorated)
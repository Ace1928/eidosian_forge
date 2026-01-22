from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_user_can_catch_Exception(self):
    case = self.make_case()
    e = Exception('Yo')

    def raises():
        raise e
    log = []
    run = RunTest(case, [(Exception, None)])
    run.result = ExtendedTestResult()
    status = run._run_user(raises)
    self.assertEqual(run.exception_caught, status)
    self.assertEqual([], run.result._events)
    self.assertEqual([], log)
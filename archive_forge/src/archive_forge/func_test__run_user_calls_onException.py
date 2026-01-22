from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_user_calls_onException(self):
    case = self.make_case()
    log = []

    def handler(exc_info):
        log.append('got it')
        self.assertEqual(3, len(exc_info))
        self.assertIsInstance(exc_info[1], KeyError)
        self.assertIs(KeyError, exc_info[0])
    case.addOnException(handler)
    e = KeyError('Yo')

    def raises():
        raise e
    run = RunTest(case, [(KeyError, None)])
    run.result = ExtendedTestResult()
    status = run._run_user(raises)
    self.assertEqual(run.exception_caught, status)
    self.assertEqual([], run.result._events)
    self.assertEqual(['got it'], log)
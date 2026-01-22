from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_prepared_result_calls_stop_test_always(self):
    result = ExtendedTestResult()
    case = self.make_case()

    def inner():
        raise Exception('foo')
    run = RunTest(case, lambda x: x)
    run._run_core = inner
    self.assertThat(lambda: run.run(result), Raises(MatchesException(Exception('foo'))))
    self.assertEqual([('startTest', case), ('stopTest', case)], result._events)
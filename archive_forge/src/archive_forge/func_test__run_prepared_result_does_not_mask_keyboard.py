from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_prepared_result_does_not_mask_keyboard(self):
    tearDownRuns = []

    class Case(TestCase):

        def test(self):
            raise KeyboardInterrupt('go')

        def _run_teardown(self, result):
            tearDownRuns.append(self)
            return super()._run_teardown(result)
    case = Case('test')
    run = RunTest(case)
    run.result = ExtendedTestResult()
    self.assertThat(lambda: run._run_prepared_result(run.result), Raises(MatchesException(KeyboardInterrupt)))
    self.assertEqual([('startTest', case), ('stopTest', case)], run.result._events)
    self.assertThat(tearDownRuns, HasLength(1))
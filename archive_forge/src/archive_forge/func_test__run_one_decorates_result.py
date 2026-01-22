from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_one_decorates_result(self):
    log = []

    class Run(RunTest):

        def _run_prepared_result(self, result):
            log.append(result)
            return result
    run = Run(self.make_case(), lambda x: x)
    result = run._run_one('foo')
    self.assertEqual([result], log)
    self.assertIsInstance(log[0], ExtendedToOriginalDecorator)
    self.assertEqual('foo', result.decorated)
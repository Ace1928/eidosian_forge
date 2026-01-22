from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__init____handlers_last_resort(self):
    handlers = [('quux', 'baz')]
    last_resort = 'foo'
    run = RunTest('bar', handlers, last_resort)
    self.assertEqual(last_resort, run.last_resort)
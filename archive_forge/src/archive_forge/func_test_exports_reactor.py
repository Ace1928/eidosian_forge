import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_exports_reactor(self):
    reactor = self.make_reactor()
    timeout = self.make_timeout()

    class SomeCase(TestCase):

        def test_cruft(self):
            self.assertIs(reactor, self.reactor)
    test = SomeCase('test_cruft')
    runner = self.make_runner(test, timeout)
    result = TestResult()
    runner.run(result)
    self.assertEqual([], result.errors)
    self.assertEqual([], result.failures)
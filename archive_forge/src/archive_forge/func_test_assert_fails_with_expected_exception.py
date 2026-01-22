import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_assert_fails_with_expected_exception(self):
    try:
        1 / 0
    except ZeroDivisionError:
        f = failure.Failure()
    d = assert_fails_with(defer.fail(f), ZeroDivisionError)
    return d.addCallback(self.assertThat, Equals(f.value))
import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_custom_failure_exception(self):

    class CustomException(Exception):
        pass
    marker = object()
    d = assert_fails_with(defer.succeed(marker), RuntimeError, failureException=CustomException)

    def check_result(failure):
        failure.trap(CustomException)
        self.assertThat(str(failure.value), Equals(f'RuntimeError not raised ({marker!r} returned)'))
    return d.addCallbacks(lambda x: self.fail('Should not have succeeded'), check_result)
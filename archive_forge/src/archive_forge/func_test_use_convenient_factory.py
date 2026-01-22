import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_use_convenient_factory(self):
    factory = AsynchronousDeferredRunTest.make_factory()

    class SomeCase(TestCase):
        run_tests_with = factory

        def test_something(self):
            pass
    case = SomeCase('test_something')
    case.run()
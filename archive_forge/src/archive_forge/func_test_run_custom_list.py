import doctest
import io
import sys
from textwrap import dedent
import unittest
from unittest import TestSuite
import testtools
from testtools import TestCase, run, skipUnless
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools import TestCase
from fixtures import Fixture
from testresources import (
from testtools import TestCase
from testtools import TestCase, clone_test_with_new_id
def test_run_custom_list(self):
    self.useFixture(SampleTestFixture())
    tests = []

    class CaptureList(run.TestToolsTestRunner):

        def list(self, test):
            tests.append({case.id() for case in testtools.testsuite.iterate_tests(test)})
    out = io.StringIO()
    try:
        program = run.TestProgram(argv=['prog', '-l', 'testtools.runexample.test_suite'], stdout=out, testRunner=CaptureList)
    except SystemExit:
        exc_info = sys.exc_info()
        raise AssertionError('-l tried to exit. %r' % exc_info[1])
    self.assertEqual([{'testtools.runexample.TestFoo.test_bar', 'testtools.runexample.TestFoo.test_quux'}], tests)
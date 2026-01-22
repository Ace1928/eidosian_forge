import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
def test_countTestCases_no_scenarios(self):

    class ReferenceTest(self.Implementation):

        def test_check_foo(self):
            pass
    test = ReferenceTest('test_check_foo')
    self.assertEqual(1, test.countTestCases())
import unittest
from testtools import (
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.testresult.doubles import (
def test_useFixture_cleanups_raise_caught(self):
    calls = []

    def raiser(ignored):
        calls.append('called')
        raise Exception('foo')
    fixture = fixtures.FunctionFixture(lambda: None, raiser)

    class SimpleTest(TestCase):

        def test_foo(self):
            self.useFixture(fixture)
    result = unittest.TestResult()
    SimpleTest('test_foo').run(result)
    self.assertFalse(result.wasSuccessful())
    self.assertEqual(['called'], calls)
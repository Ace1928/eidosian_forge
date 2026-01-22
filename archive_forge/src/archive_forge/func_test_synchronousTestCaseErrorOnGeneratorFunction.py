import gc
import sys
import unittest as pyunit
import weakref
from io import StringIO
from twisted.internet import defer, reactor
from twisted.python.compat import _PYPY
from twisted.python.reflect import namedAny
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import (
from twisted.trial.test import erroneous
from twisted.trial.test.test_suppression import SuppressionMixin
def test_synchronousTestCaseErrorOnGeneratorFunction(self):
    """
        In a SynchronousTestCase, a test method which is a generator function
        is reported as an error, as such a method will never run assertions.
        """

    class GeneratorSynchronousTestCase(unittest.SynchronousTestCase):
        """
            A fake SynchronousTestCase for testing purposes.
            """

        def test_generator(self):
            """
                A method which is also a generator function, for testing
                purposes.
                """
            self.fail('this should never be reached')
            yield
    testCase = GeneratorSynchronousTestCase('test_generator')
    result = reporter.TestResult()
    testCase.run(result)
    self.assertEqual(len(result.failures), 0)
    self.assertEqual(len(result.errors), 1)
    self.assertIn('GeneratorSynchronousTestCase.test_generator', result.errors[0][1].value.args[0])
    self.assertIn('GeneratorSynchronousTestCase testMethod=test_generator', result.errors[0][1].value.args[0])
    self.assertIn('is a generator function and therefore will never run', result.errors[0][1].value.args[0])
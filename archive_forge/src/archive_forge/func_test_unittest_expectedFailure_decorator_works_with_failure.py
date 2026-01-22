from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_unittest_expectedFailure_decorator_works_with_failure(self):

    class ReferenceTest(TestCase):

        @unittest.expectedFailure
        def test_fails_expectedly(self):
            self.assertEqual(1, 0)
    test = ReferenceTest('test_fails_expectedly')
    result = test.run()
    self.assertEqual(True, result.wasSuccessful())
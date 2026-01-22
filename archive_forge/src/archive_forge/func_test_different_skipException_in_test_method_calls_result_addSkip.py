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
def test_different_skipException_in_test_method_calls_result_addSkip(self):

    class SkippingTest(TestCase):
        skipException = ValueError

        def test_that_raises_skipException(self):
            self.skipTest('skipping this test')
    events = []
    result = Python27TestResult(events)
    test = SkippingTest('test_that_raises_skipException')
    test.run(result)
    case = result._events[0][1]
    self.assertEqual([('startTest', case), ('addSkip', case, 'skipping this test'), ('stopTest', case)], events)
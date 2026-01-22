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
def test_skip__in_setup_with_old_result_object_calls_addSuccess(self):

    class SkippingTest(TestCase):

        def setUp(self):
            TestCase.setUp(self)
            raise self.skipException('skipping this test')

        def test_that_raises_skipException(self):
            pass
    events = []
    result = Python26TestResult(events)
    test = SkippingTest('test_that_raises_skipException')
    test.run(result)
    self.assertEqual('addSuccess', events[1][0])
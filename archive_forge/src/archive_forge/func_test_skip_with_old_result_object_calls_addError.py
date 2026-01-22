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
def test_skip_with_old_result_object_calls_addError(self):

    class SkippingTest(TestCase):

        def test_that_raises_skipException(self):
            raise self.skipException('skipping this test')
    events = []
    result = Python26TestResult(events)
    test = SkippingTest('test_that_raises_skipException')
    test.run(result)
    self.assertEqual('addSuccess', events[1][0])
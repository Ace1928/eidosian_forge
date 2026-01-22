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
def test_skipIf_decorator(self):

    class SkippingTest(TestCase):

        @skipIf(True, 'skipping this test')
        def test_that_is_decorated_with_skipIf(self):
            self.fail()
    events = []
    result = Python26TestResult(events)
    test = SkippingTest('test_that_is_decorated_with_skipIf')
    test.run(result)
    self.assertEqual('addSuccess', events[1][0])
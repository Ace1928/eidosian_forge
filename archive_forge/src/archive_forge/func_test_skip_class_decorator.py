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
def test_skip_class_decorator(self):

    @skip('skipping this testcase')
    class SkippingTest(TestCase):

        def test_that_is_decorated_with_skip(self):
            self.fail()
    events = []
    result = Python26TestResult(events)
    try:
        test = SkippingTest('test_that_is_decorated_with_skip')
    except TestSkipped:
        self.fail('TestSkipped raised')
    test.run(result)
    self.assertEqual('addSuccess', events[1][0])
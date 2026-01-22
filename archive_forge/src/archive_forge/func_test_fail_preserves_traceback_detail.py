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
def test_fail_preserves_traceback_detail(self):

    class Test(TestCase):

        def test(self):
            self.addDetail('traceback', text_content('foo'))
            self.fail('bar')
    test = Test('test')
    result = ExtendedTestResult()
    test.run(result)
    self.assertEqual({'traceback', 'traceback-1'}, set(result._events[1][2].keys()))
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
def test_tearDownNotCalled(self):

    class DoesnotcalltearDown(TestCase):

        def test_method(self):
            pass

        def tearDown(self):
            pass
    result = unittest.TestResult()
    DoesnotcalltearDown('test_method').run(result)
    self.assertThat(result.errors, HasLength(1))
    self.assertThat(result.errors[0][1], DocTestMatches('...ValueError...File...testtools/tests/test_testcase.py...', ELLIPSIS))
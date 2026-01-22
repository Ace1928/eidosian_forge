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
def test_raising__UnexpectedSuccess_py27(self):
    case = self.make_unexpected_case()
    result = Python27TestResult()
    case.run(result)
    case = result._events[0][1]
    self.assertEqual([('startTest', case), ('addUnexpectedSuccess', case), ('stopTest', case)], result._events)
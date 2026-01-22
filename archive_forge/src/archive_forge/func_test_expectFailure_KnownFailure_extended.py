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
def test_expectFailure_KnownFailure_extended(self):
    case = self.make_xfail_case_xfails()
    self.assertDetailsProvided(case, 'addExpectedFailure', ['foo', 'traceback', 'reason'])
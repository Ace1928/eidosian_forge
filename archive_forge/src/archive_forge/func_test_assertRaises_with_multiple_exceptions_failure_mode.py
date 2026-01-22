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
def test_assertRaises_with_multiple_exceptions_failure_mode(self):
    expectedExceptions = (RuntimeError, ZeroDivisionError)
    self.assertRaises(self.failureException, self.assertRaises, expectedExceptions, lambda: None)
    self.assertFails('<function ...<lambda> at ...> returned None', self.assertRaises, expectedExceptions, lambda: None)
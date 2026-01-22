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
def check_test_does_not_run_setup(self, test, reason):
    result = test.run()
    self.assertTrue(result.wasSuccessful())
    self.assertIn(reason, result.skip_reasons, result.skip_reasons)
    self.assertFalse(test.setup_ran)
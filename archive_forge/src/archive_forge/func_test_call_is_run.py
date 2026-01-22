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
def test_call_is_run(self):
    test = self.makePlaceHolder()
    run_log = []
    test.run(LoggingResult(run_log))
    call_log = []
    test(LoggingResult(call_log))
    self.assertEqual(run_log, call_log)
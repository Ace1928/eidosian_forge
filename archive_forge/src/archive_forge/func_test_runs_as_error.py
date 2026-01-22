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
def test_runs_as_error(self):
    error = self.makeException()
    test = self.makePlaceHolder(error=error)
    result = ExtendedTestResult()
    log = result._events
    test.run(result)
    self.assertEqual([('tags', set(), set()), ('startTest', test), ('addError', test, test._details), ('stopTest', test), ('tags', set(), set())], log)
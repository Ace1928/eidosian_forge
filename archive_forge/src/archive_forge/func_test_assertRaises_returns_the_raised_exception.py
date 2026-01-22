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
def test_assertRaises_returns_the_raised_exception(self):
    raisedExceptions = []

    def raiseError():
        try:
            raise RuntimeError('Deliberate error')
        except RuntimeError:
            raisedExceptions.append(sys.exc_info()[1])
            raise
    exception = self.assertRaises(RuntimeError, raiseError)
    self.assertEqual(1, len(raisedExceptions))
    self.assertIs(exception, raisedExceptions[0], '{!r} is not {!r}'.format(exception, raisedExceptions[0]))
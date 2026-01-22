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
def test_assertEqual_nice_formatting(self):
    message = 'These things ought not be equal.'
    a = ['apple', 'banana', 'cherry']
    b = {'Thatcher': 'One who mends roofs of straw', 'Major': 'A military officer, ranked below colonel', 'Blair': 'To shout loudly', 'Brown': 'The colour of healthy human faeces'}
    expected_error = '\n'.join(['!=:', 'reference = %s' % pformat(a), 'actual    = %s' % pformat(b), ': ' + message])
    self.assertFails(expected_error, self.assertEqual, a, b, message)
    self.assertFails(expected_error, self.assertEquals, a, b, message)
    self.assertFails(expected_error, self.failUnlessEqual, a, b, message)
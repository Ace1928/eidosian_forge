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
def test_assertThat_verbose_unicode(self):
    matchee = '§'
    matcher = Equals('a')
    expected = 'Match failed. Matchee: %s\nMatcher: %s\nDifference: %s\n\n' % (repr(matchee).replace('\\xa7', matchee), matcher, matcher.match(matchee).describe())
    e = self.assertRaises(self.failureException, self.assertThat, matchee, matcher, verbose=True)
    self.assertEqual(expected, self.get_error_string(e))
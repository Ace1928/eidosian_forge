import doctest
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._doctest import DocTestMatches
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test___init__flags(self):
    matcher = DocTestMatches('bar\n', doctest.ELLIPSIS)
    self.assertEqual('bar\n', matcher.want)
    self.assertEqual(doctest.ELLIPSIS, matcher.flags)
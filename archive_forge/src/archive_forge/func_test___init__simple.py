import doctest
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._doctest import DocTestMatches
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test___init__simple(self):
    matcher = DocTestMatches('foo')
    self.assertEqual('foo\n', matcher.want)
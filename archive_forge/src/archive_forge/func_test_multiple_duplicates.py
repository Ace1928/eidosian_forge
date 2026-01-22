import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
def test_multiple_duplicates(self):
    a = PlaceHolder('a')
    b = PlaceHolder('b')
    c = PlaceHolder('a')
    d = PlaceHolder('b')
    error = self.assertRaises(ValueError, sorted_tests, unittest.TestSuite([a, b, c, d]))
    self.assertThat(str(error), Equals('Duplicate test ids detected: {}'.format(pformat({'a': 2, 'b': 2}))))
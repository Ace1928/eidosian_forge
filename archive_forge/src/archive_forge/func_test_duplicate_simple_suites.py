import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
def test_duplicate_simple_suites(self):
    a = PlaceHolder('a')
    b = PlaceHolder('b')
    c = PlaceHolder('a')
    self.assertRaises(ValueError, sorted_tests, unittest.TestSuite([a, b, c]))
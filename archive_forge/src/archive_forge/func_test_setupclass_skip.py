import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
def test_setupclass_skip(self):

    class Skips(TestCase):

        @classmethod
        def setUpClass(cls):
            raise cls.skipException('foo')

        def test_notrun(self):
            pass
    suite = unittest.TestSuite([Skips('test_notrun')])
    log = []
    result = LoggingResult(log)
    suite.run(result)
    self.assertEqual(['addSkip'], [item[0] for item in log])
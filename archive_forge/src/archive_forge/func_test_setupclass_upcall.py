import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
def test_setupclass_upcall(self):

    class Simples(TestCase):

        @classmethod
        def setUpClass(cls):
            super().setUpClass()

        def test_simple(self):
            pass
    suite = unittest.TestSuite([Simples('test_simple')])
    log = []
    result = LoggingResult(log)
    suite.run(result)
    self.assertEqual(['startTest', 'addSuccess', 'stopTest'], [item[0] for item in log])
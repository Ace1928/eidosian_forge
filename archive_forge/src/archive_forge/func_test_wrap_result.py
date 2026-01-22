import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
def test_wrap_result(self):
    wrap_log = []

    def wrap_result(thread_safe_result, thread_number):
        wrap_log.append((thread_safe_result.result.decorated, thread_number))
        return thread_safe_result
    result_log = []
    result = LoggingResult(result_log)
    test1 = Sample('test_method1')
    test2 = Sample('test_method2')
    original_suite = unittest.TestSuite([test1, test2])
    suite = ConcurrentTestSuite(original_suite, self.split_suite, wrap_result=wrap_result)
    suite.run(result)
    self.assertEqual([(result, 0), (result, 1)], wrap_log)
    self.assertNotEqual([], result_log)
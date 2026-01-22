import subprocess
import sys
import unittest
from datetime import datetime
from io import BytesIO
from testtools import TestCase
from testtools.compat import _b
from testtools.testresult.doubles import ExtendedTestResult, StreamResult
import iso8601
import subunit
from subunit.test_results import make_tag_filter, TestResultFilter
from subunit import ByteStreamToStreamResult, StreamResultToBytes
def test_include_success(self):
    """Successes can be included if requested."""
    filtered_result = unittest.TestResult()
    result_filter = TestResultFilter(filtered_result, filter_success=False)
    self.run_tests(result_filter)
    self.assertEqual(['error'], [error[0].id() for error in filtered_result.errors])
    self.assertEqual(['failed'], [failure[0].id() for failure in filtered_result.failures])
    self.assertEqual(5, filtered_result.testsRun)
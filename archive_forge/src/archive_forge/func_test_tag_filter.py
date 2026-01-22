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
def test_tag_filter(self):
    tag_filter = make_tag_filter(['global'], ['local'])
    result = ExtendedTestResult()
    result_filter = TestResultFilter(result, filter_success=False, filter_predicate=tag_filter)
    self.run_tests(result_filter)
    tests_included = [event[1] for event in result._events if event[0] == 'startTest']
    tests_expected = list(map(subunit.RemotedTestCase, ['passed', 'error', 'skipped', 'todo']))
    self.assertEqual(tests_expected, tests_included)
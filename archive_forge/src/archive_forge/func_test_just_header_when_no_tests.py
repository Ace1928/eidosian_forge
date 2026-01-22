import csv
import datetime
import sys
import unittest
from io import StringIO
import testtools
from testtools import TestCase
from testtools.content import TracebackContent, text_content
from testtools.testresult.doubles import ExtendedTestResult
import subunit
import iso8601
import subunit.test_results
def test_just_header_when_no_tests(self):
    stream = StringIO()
    result = subunit.test_results.CsvResult(stream)
    result.startTestRun()
    result.stopTestRun()
    self.assertEqual([['test', 'status', 'start_time', 'stop_time']], self.parse_stream(stream))
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
def test_calling_time_inhibits_automatic_time(self):
    time = datetime.datetime(2009, 10, 11, 12, 13, 14, 15, iso8601.UTC)
    self.result.time(time)
    self.result.startTest(self)
    self.result.stopTest(self)
    self.assertEqual(1, len(self.decorated._calls))
    self.assertEqual(time, self.decorated._calls[0])
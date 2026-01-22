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
def test_add_xfail(self):
    self.result.startTest(self)
    try:
        1 / 0
    except ZeroDivisionError:
        error = sys.exc_info()
    self.result.addExpectedFailure(self, error)
    self.result.stopTest(self)
    self.assertCalled(status='xfail', details={'traceback': TracebackContent(error, self)})
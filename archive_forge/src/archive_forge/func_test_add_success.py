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
def test_add_success(self):
    self.result.startTest(self)
    self.result.addSuccess(self)
    self.result.stopTest(self)
    self.assertCalled(status='success')
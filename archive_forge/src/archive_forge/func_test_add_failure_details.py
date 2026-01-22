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
def test_add_failure_details(self):
    self.result.startTest(self)
    details = {'foo': text_content('bar')}
    self.result.addFailure(self, details=details)
    self.result.stopTest(self)
    self.assertCalled(status='failure', details=details)
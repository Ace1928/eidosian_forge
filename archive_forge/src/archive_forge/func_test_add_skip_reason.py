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
def test_add_skip_reason(self):
    self.result.startTest(self)
    reason = self.getUniqueString()
    self.result.addSkip(self, reason)
    self.result.stopTest(self)
    self.assertCalled(status='skip', details={'reason': text_content(reason)})
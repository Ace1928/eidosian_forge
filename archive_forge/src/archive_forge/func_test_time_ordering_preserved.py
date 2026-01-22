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
def test_time_ordering_preserved(self):
    date_a = datetime(year=2000, month=1, day=1, tzinfo=iso8601.UTC)
    date_b = datetime(year=2000, month=1, day=2, tzinfo=iso8601.UTC)
    date_c = datetime(year=2000, month=1, day=3, tzinfo=iso8601.UTC)
    subunit_stream = _b('\n'.join(['time: %s', 'test: foo', 'time: %s', 'error: foo', 'time: %s', '']) % (date_a, date_b, date_c))
    result = ExtendedTestResult()
    result_filter = TestResultFilter(result)
    self.run_tests(result_filter, subunit_stream)
    foo = subunit.RemotedTestCase('foo')
    self.maxDiff = None
    self.assertEqual([('time', date_a), ('time', date_b), ('startTest', foo), ('addError', foo, {}), ('stopTest', foo), ('time', date_c)], result._events)
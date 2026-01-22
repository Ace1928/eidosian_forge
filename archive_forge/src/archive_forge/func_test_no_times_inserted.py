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
def test_no_times_inserted(self):
    result = ExtendedTestResult()
    tag_collapser = subunit.test_results.TimeCollapsingDecorator(result)
    a_time = self.make_time()
    tag_collapser.time(a_time)
    foo = subunit.RemotedTestCase('foo')
    tag_collapser.startTest(foo)
    tag_collapser.addSuccess(foo)
    tag_collapser.stopTest(foo)
    self.assertEqual([('time', a_time), ('startTest', foo), ('addSuccess', foo), ('stopTest', foo)], result._events)
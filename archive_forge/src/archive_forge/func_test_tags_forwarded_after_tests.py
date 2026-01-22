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
def test_tags_forwarded_after_tests(self):
    test = subunit.RemotedTestCase('foo')
    result = ExtendedTestResult()
    tag_collapser = subunit.test_results.TagCollapsingDecorator(result)
    tag_collapser.startTestRun()
    tag_collapser.startTest(test)
    tag_collapser.addSuccess(test)
    tag_collapser.stopTest(test)
    tag_collapser.tags({'a'}, {'b'})
    tag_collapser.stopTestRun()
    self.assertEqual([('startTestRun',), ('startTest', test), ('addSuccess', test), ('stopTest', test), ('tags', {'a'}, {'b'}), ('stopTestRun',)], result._events)
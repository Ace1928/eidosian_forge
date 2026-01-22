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
def test_tags_collapsed_inside_of_tests_different_ordering(self):
    result = ExtendedTestResult()
    tag_collapser = subunit.test_results.TagCollapsingDecorator(result)
    test = subunit.RemotedTestCase('foo')
    tag_collapser.startTest(test)
    tag_collapser.tags(set(), {'a'})
    tag_collapser.tags({'a', 'b'}, set())
    tag_collapser.tags({'c'}, set())
    tag_collapser.stopTest(test)
    self.assertEqual([('startTest', test), ('tags', {'a', 'b', 'c'}, set()), ('stopTest', test)], result._events)
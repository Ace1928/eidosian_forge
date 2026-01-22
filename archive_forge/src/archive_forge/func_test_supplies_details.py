from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_supplies_details(self):
    details = {'quux': None}
    test = PlaceHolder('foo', details=details)
    result = ExtendedTestResult()
    test.run(result)
    self.assertEqual([('tags', set(), set()), ('startTest', test), ('addSuccess', test, details), ('stopTest', test), ('tags', set(), set())], result._events)
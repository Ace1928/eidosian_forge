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
def test_clone_test_with_new_id(self):

    class FooTestCase(TestCase):

        def test_foo(self):
            pass
    test = FooTestCase('test_foo')
    oldName = test.id()
    newName = self.getUniqueString()
    newTest = clone_test_with_new_id(test, newName)
    self.assertEqual(newName, newTest.id())
    self.assertEqual(oldName, test.id(), 'the original test instance should be unchanged.')
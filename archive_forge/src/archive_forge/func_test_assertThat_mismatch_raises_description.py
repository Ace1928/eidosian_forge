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
def test_assertThat_mismatch_raises_description(self):
    calls = []

    class Mismatch:

        def __init__(self, thing):
            self.thing = thing

        def describe(self):
            calls.append(('describe_diff', self.thing))
            return 'object is not a thing'

        def get_details(self):
            return {}

    class Matcher:

        def match(self, thing):
            calls.append(('match', thing))
            return Mismatch(thing)

        def __str__(self):
            calls.append(('__str__',))
            return 'a description'

    class Test(TestCase):

        def test(self):
            self.assertThat('foo', Matcher())
    result = Test('test').run()
    self.assertEqual([('match', 'foo'), ('describe_diff', 'foo')], calls)
    self.assertFalse(result.wasSuccessful())
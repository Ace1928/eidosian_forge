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
def test_multiple_addDetails_from_Mismatch(self):
    content = self.get_content()

    class Mismatch:

        def describe(self):
            return 'Mismatch'

        def get_details(self):
            return {'foo': content, 'bar': content}

    class Matcher:

        def match(self, thing):
            return Mismatch()

        def __str__(self):
            return 'a description'

    class Case(TestCase):

        def test(self):
            self.assertThat('foo', Matcher())
    self.assertDetailsProvided(Case('test'), 'addFailure', ['bar', 'foo', 'traceback'])
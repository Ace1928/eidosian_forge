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
def make_xfail_case_succeeds(self):
    content = self.get_content()

    class Case(TestCase):

        def test(self):
            self.addDetail('foo', content)
            self.expectFailure('we are sad', self.assertEqual, 1, 1)
    case = Case('test')
    return case
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
def make_unexpected_case(self):

    class Case(TestCase):

        def test(self):
            raise testcase._UnexpectedSuccess
    case = Case('test')
    return case
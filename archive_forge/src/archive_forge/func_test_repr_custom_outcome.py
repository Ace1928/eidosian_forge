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
def test_repr_custom_outcome(self):
    test = PlaceHolder('test id', outcome='addSkip')
    self.assertEqual("<testtools.testcase.PlaceHolder('addSkip', %r, {})>" % test.id(), repr(test))
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
def test_getUniqueString_prefix(self):
    name_one = self.getUniqueString('foo')
    self.assertThat(name_one, Equals('foo-1'))
    name_two = self.getUniqueString('bar')
    self.assertThat(name_two, Equals('bar-2'))
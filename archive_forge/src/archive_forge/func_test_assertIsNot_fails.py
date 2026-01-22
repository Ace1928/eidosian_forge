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
def test_assertIsNot_fails(self):
    self.assertFails('None matches Is(None)', self.assertIsNot, None, None)
    some_list = [42]
    self.assertFails('[42] matches Is([42])', self.assertIsNot, some_list, some_list)
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
def test_keyboard_interrupt_not_caught(self):
    test = make_test_case(self.getUniqueString(), cleanups=[lambda _: raise_(KeyboardInterrupt())])
    self.assertThat(test.run, Raises(MatchesException(KeyboardInterrupt)))
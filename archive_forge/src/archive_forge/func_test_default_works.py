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
def test_default_works(self):
    events = []

    class Case(TestCase):

        def method(self):
            self.onException(an_exc_info)
            events.append(True)
    case = Case('method')
    case.run()
    self.assertThat(events, Equals([True]))
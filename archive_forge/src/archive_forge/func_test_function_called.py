import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_function_called(self):
    calls = []
    marker = object()
    self.make_spinner().run(self.make_timeout(), calls.append, marker)
    self.assertThat(calls, Equals([marker]))
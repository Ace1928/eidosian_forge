import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_return_value_returned(self):
    marker = object()
    result = self.make_spinner().run(self.make_timeout(), lambda: marker)
    self.assertThat(result, Is(marker))
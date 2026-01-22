import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_exception_reraised(self):
    self.assertThat(lambda: self.make_spinner().run(self.make_timeout(), lambda: 1 / 0), Raises(MatchesException(ZeroDivisionError)))
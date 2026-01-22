import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_not_reentrant(self):
    spinner = self.make_spinner()
    self.assertThat(lambda: spinner.run(self.make_timeout(), spinner.run, self.make_timeout(), lambda: None), Raises(MatchesException(_spinner.ReentryError)))
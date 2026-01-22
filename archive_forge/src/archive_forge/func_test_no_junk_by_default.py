import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_no_junk_by_default(self):
    spinner = self.make_spinner()
    self.assertThat(spinner.get_junk(), Equals([]))
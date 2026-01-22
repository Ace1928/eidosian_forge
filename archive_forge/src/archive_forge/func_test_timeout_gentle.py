import signal
import time
import testtools
from testtools.testcase import (
from testtools.matchers import raises
import fixtures
def test_timeout_gentle(self):
    self.requireUnix()
    self.assertRaises(fixtures.TimeoutException, sample_long_delay_with_gentle_timeout)
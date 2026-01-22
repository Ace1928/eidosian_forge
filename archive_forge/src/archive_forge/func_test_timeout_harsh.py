import signal
import time
import testtools
from testtools.testcase import (
from testtools.matchers import raises
import fixtures
def test_timeout_harsh(self):
    self.requireUnix()

    class GotAlarm(Exception):
        pass

    def sigalrm_handler(signum, frame):
        raise GotAlarm()
    old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
    self.addCleanup(signal.signal, signal.SIGALRM, old_handler)
    self.assertThat(sample_long_delay_with_harsh_timeout, raises(GotAlarm))
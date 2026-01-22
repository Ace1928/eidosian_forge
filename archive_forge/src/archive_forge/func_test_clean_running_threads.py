import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_clean_running_threads(self):
    import threading
    import time
    current_threads = list(threading.enumerate())
    reactor = self.make_reactor()
    timeout = self.make_timeout()
    spinner = self.make_spinner(reactor)
    spinner.run(timeout, reactor.callInThread, time.sleep, timeout / 2.0)
    self.assertThat(list(threading.enumerate()), Equals(current_threads))
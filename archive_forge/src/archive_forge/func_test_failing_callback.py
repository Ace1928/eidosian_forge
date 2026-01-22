import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def test_failing_callback(self):
    calls = []

    def call_me():
        calls.append('called')

    def fail_me():
        raise RuntimeError('something bad happened')
    signals.register_on_hangup('myid', call_me)
    signals.register_on_hangup('otherid', fail_me)
    signals._sighup_handler(SIGHUP, None)
    signals.unregister_on_hangup('myid')
    signals.unregister_on_hangup('otherid')
    log = self.get_log()
    self.assertContainsRe(log, '(?s)Traceback.*RuntimeError')
    self.assertEqual(['called'], calls)
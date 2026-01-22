import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def test_registered_callback_gets_called(self):
    calls = []

    def call_me():
        calls.append('called')
    signals.register_on_hangup('myid', call_me)
    signals._sighup_handler(SIGHUP, None)
    self.assertEqual(['called'], calls)
    signals.unregister_on_hangup('myid')
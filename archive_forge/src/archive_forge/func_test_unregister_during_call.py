import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def test_unregister_during_call(self):
    calls = []

    def call_me_and_unregister():
        signals.unregister_on_hangup('myid')
        calls.append('called_and_unregistered')

    def call_me():
        calls.append('called')
    signals.register_on_hangup('myid', call_me_and_unregister)
    signals.register_on_hangup('other', call_me)
    signals._sighup_handler(SIGHUP, None)
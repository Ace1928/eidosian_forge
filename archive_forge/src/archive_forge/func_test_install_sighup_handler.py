import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def test_install_sighup_handler(self):
    signals._on_sighup = None
    orig = signals.install_sighup_handler()
    if getattr(signal, 'SIGHUP', None) is not None:
        cur = signal.getsignal(SIGHUP)
        self.assertEqual(signals._sighup_handler, cur)
    self.assertIsNot(None, signals._on_sighup)
    signals.restore_sighup_handler(orig)
    self.assertIs(None, signals._on_sighup)
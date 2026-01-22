import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def test_unregister_not_present(self):
    signals.unregister_on_hangup('no-such-id')
    log = self.get_log()
    self.assertContainsRe(log, 'Error occurred during unregister_on_hangup:')
    self.assertContainsRe(log, '(?s)Traceback.*KeyError')
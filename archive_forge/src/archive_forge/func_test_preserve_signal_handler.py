import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_preserve_signal_handler(self):
    signals = ['SIGINT', 'SIGTERM', 'SIGCHLD']
    signals = list(filter(None, (getattr(signal, name, None) for name in signals)))
    for sig in signals:
        self.addCleanup(signal.signal, sig, signal.getsignal(sig))
    new_hdlrs = list((lambda *a: None for _ in signals))
    for sig, hdlr in zip(signals, new_hdlrs):
        signal.signal(sig, hdlr)
    spinner = self.make_spinner()
    spinner.run(self.make_timeout(), lambda: None)
    self.assertItemsEqual(new_hdlrs, list(map(signal.getsignal, signals)))
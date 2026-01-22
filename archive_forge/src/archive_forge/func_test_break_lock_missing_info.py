import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_break_lock_missing_info(self):
    """break_lock works even if the info file is missing (and tells the UI
        that it is corrupt).
        """
    ld = self.get_lock()
    ld2 = self.get_lock()
    ld.create()
    ld.lock_write()
    ld.transport.delete('test_lock/held/info')

    class LoggingUIFactory(breezy.ui.SilentUIFactory):

        def __init__(self):
            self.prompts = []

        def get_boolean(self, prompt):
            self.prompts.append(('boolean', prompt))
            return True
    ui = LoggingUIFactory()
    orig_factory = breezy.ui.ui_factory
    breezy.ui.ui_factory = ui
    try:
        ld2.break_lock()
        self.assertRaises(LockBroken, ld.unlock)
        self.assertLength(0, ui.prompts)
    finally:
        breezy.ui.ui_factory = orig_factory
    del self._lock_actions[:]
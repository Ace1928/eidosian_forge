import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
def test_second_lock_write_returns_same_token(self):
    first_token = self.lockable.lock_write()
    try:
        if first_token is None:
            return
        second_token = self.lockable.lock_write()
        try:
            self.assertEqual(first_token, second_token)
        finally:
            self.lockable.unlock()
    finally:
        self.lockable.unlock()
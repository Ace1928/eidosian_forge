from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def test_lock_not_reentrant(self):
    l = DummyLock()
    l.lock_read()
    self.assertRaises(LockError, l.lock_read)
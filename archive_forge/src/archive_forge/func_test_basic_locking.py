from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def test_basic_locking(self):
    real_lock = DummyLock()
    self.assertFalse(real_lock.is_locked())
    real_lock.lock_read()
    self.assertTrue(real_lock.is_locked())
    real_lock.unlock()
    self.assertFalse(real_lock.is_locked())
    result = real_lock.lock_write()
    self.assertEqual('token', result)
    self.assertTrue(real_lock.is_locked())
    real_lock.unlock()
    self.assertFalse(real_lock.is_locked())
    self.assertEqual(['lock_read', 'unlock', 'lock_write', 'unlock'], real_lock._calls)
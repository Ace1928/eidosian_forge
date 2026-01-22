from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def test_reenter_with_token(self):
    real_lock = DummyLock()
    l1 = CountedLock(real_lock)
    l2 = CountedLock(real_lock)
    token = l1.lock_write()
    self.assertEqual('token', token)
    del l1
    self.assertTrue(real_lock.is_locked())
    self.assertFalse(l2.is_locked())
    self.assertEqual(token, l2.lock_write(token=token))
    self.assertTrue(l2.is_locked())
    self.assertTrue(real_lock.is_locked())
    l2.unlock()
    self.assertFalse(l2.is_locked())
    self.assertFalse(real_lock.is_locked())
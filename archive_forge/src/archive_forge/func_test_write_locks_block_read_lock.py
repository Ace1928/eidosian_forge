from .. import debug, errors, lock, tests
from .scenarios import load_tests_apply_scenarios
def test_write_locks_block_read_lock(self):
    w_lock = self.write_lock('a-lock-file')
    try:
        if lock.have_fcntl and self.read_lock is lock._fcntl_ReadLock:
            debug.debug_flags.add('strict_locks')
            self.assertRaises(errors.LockContention, self.read_lock, 'a-lock-file')
            debug.debug_flags.remove('strict_locks')
            try:
                r_lock = self.read_lock('a-lock-file')
            except errors.LockContention:
                self.fail('Unexpected success. fcntl write locks do not usually block read locks')
            else:
                r_lock.unlock()
                self.knownFailure("fcntl write locks don't block read locks without -Dlock")
        else:
            self.assertRaises(errors.LockContention, self.read_lock, 'a-lock-file')
    finally:
        w_lock.unlock()
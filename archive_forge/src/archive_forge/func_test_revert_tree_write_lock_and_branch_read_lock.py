import os
from breezy import branch, builtins, errors, lock
from breezy.tests import TestCaseInTempDir, transport_util
def test_revert_tree_write_lock_and_branch_read_lock(self):
    locks_acquired = []
    locks_released = []
    lock.Lock.hooks.install_named_hook('lock_acquired', locks_acquired.append, None)
    lock.Lock.hooks.install_named_hook('lock_released', locks_released.append, None)
    revert = builtins.cmd_revert()
    revert.run()
    self.assertLength(1, locks_acquired)
    self.assertLength(1, locks_released)
    self.assertEqual(locks_acquired[0].details, locks_released[0].details)
    self.assertEndsWith(locks_acquired[0].lock_url, '/checkout/lock')
    self.assertEndsWith(locks_released[0].lock_url, '/checkout/lock')
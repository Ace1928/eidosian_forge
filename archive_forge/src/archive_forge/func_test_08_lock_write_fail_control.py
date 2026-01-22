from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_08_lock_write_fail_control(self):
    b = self.get_instrumented_branch()
    b.control_files.disable_lock_write()
    self.assertRaises(lock_helpers.TestPreventLocking, b.lock_write)
    self.assertFalse(b.is_locked())
    self.assertFalse(b.repository.is_locked())
    if self.combined_control:
        self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('rc', 'lw', True), ('bc', 'lw', False), ('r', 'ul', True), ('rc', 'ul', True)], self.locks)
    else:
        self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('bc', 'lw', False), ('r', 'ul', True)], self.locks)
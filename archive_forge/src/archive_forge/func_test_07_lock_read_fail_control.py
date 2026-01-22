from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_07_lock_read_fail_control(self):
    b = self.get_instrumented_branch()
    b.control_files.disable_lock_read()
    self.assertRaises(lock_helpers.TestPreventLocking, b.lock_read)
    self.assertFalse(b.is_locked())
    self.assertFalse(b.repository.is_locked())
    if self.combined_control:
        self.assertEqual([('b', 'lr', True), ('r', 'lr', True), ('rc', 'lr', True), ('bc', 'lr', False), ('r', 'ul', True), ('rc', 'ul', True)], self.locks)
    else:
        self.assertEqual([('b', 'lr', True), ('r', 'lr', True), ('bc', 'lr', False), ('r', 'ul', True)], self.locks)
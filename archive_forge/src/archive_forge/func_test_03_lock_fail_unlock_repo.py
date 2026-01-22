from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_03_lock_fail_unlock_repo(self):
    b = self.get_instrumented_branch()
    b.repository.disable_unlock()
    self.assertFalse(b.is_locked())
    self.assertFalse(b.repository.is_locked())
    b.lock_write()
    try:
        self.assertTrue(b.is_locked())
        self.assertTrue(b.repository.is_locked())
        self.assertLogsError(lock_helpers.TestPreventLocking, b.unlock)
        if self.combined_control:
            self.assertTrue(b.is_locked())
        else:
            self.assertFalse(b.is_locked())
        self.assertTrue(b.repository.is_locked())
        if self.combined_control:
            self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('rc', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', False)], self.locks)
        else:
            self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', False)], self.locks)
    finally:
        b.repository._other.unlock()
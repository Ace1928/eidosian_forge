from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def test_read_unlock_write(self):
    """Make sure that unlocking allows us to lock write"""
    a_lock = self.read_lock('a-file')
    a_lock.unlock()
    a_lock = self.write_lock('a-file')
    a_lock.unlock()
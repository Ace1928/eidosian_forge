from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def test_readonly_file(self):
    """If the file is readonly, we can take a read lock.

        But we shouldn't be able to take a write lock.
        """
    self.requireFeature(features.not_running_as_root)
    osutils.make_readonly('a-file')
    self.assertRaises(IOError, open, 'a-file', 'rb+')
    a_lock = self.read_lock('a-file')
    a_lock.unlock()
    self.assertRaises(errors.LockFailed, self.write_lock, 'a-file')
from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def test_create_if_needed_write(self):
    """We will create the file if it doesn't exist yet."""
    a_lock = self.write_lock('other-file')
    self.addCleanup(a_lock.unlock)
    txt = a_lock.f.read()
    self.assertEqual(b'', txt)
    a_lock.f.seek(0)
    a_lock.f.write(b'foo\n')
    a_lock.f.seek(0)
    txt = a_lock.f.read()
    self.assertEqual(b'foo\n', txt)
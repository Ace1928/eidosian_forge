from breezy import workingtree
from breezy.tests import TestCaseWithTransport
def break_dirstate(self, tree, completely=False):
    """Write garbage into the dirstate file."""
    self.assertIsNot(None, getattr(tree, 'current_dirstate', None))
    with tree.lock_read():
        dirstate = tree.current_dirstate()
        dirstate_path = dirstate._filename
        self.assertPathExists(dirstate_path)
    if completely:
        f = open(dirstate_path, 'wb')
    else:
        f = open(dirstate_path, 'ab')
    try:
        f.write(b'garbage-at-end-of-file\n')
    finally:
        f.close()
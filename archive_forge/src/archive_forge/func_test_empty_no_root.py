from breezy import revision
from breezy.tests import TestCaseWithTransport
from breezy.tree import FileTimestampUnavailable
def test_empty_no_root(self):
    null_tree = self.t.branch.repository.revision_tree(revision.NULL_REVISION)
    self.assertIs(None, null_tree.path2id(''))
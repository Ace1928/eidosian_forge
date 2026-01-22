from breezy import revision
from breezy.tests import TestCaseWithTransport
from breezy.tree import FileTimestampUnavailable
def test_get_file_mtime_ghost(self):
    path = next(iter(self.rev_tree.all_versioned_paths()))
    self.rev_tree.root_inventory.get_entry(self.rev_tree.path2id(path)).revision = b'ghostrev'
    self.assertRaises(FileTimestampUnavailable, self.rev_tree.get_file_mtime, path)
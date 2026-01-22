import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_guess_renames_handles_directories(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree(['tree/dir/', 'tree/dir/file'])
    tree.add(['dir', 'dir/file'], ids=[b'dir-id', b'file-id'])
    tree.commit('Added file')
    os.rename('tree/dir', 'tree/dir2')
    RenameMap.guess_renames(tree.basis_tree(), tree)
    self.assertEqual('dir2/file', tree.id2path(b'file-id'))
    self.assertEqual('dir2', tree.id2path(b'dir-id'))
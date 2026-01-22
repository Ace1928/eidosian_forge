import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_guess_renames_handles_grandparent_directories(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree(['tree/topdir/', 'tree/topdir/middledir/', 'tree/topdir/middledir/file'])
    tree.add(['topdir', 'topdir/middledir', 'topdir/middledir/file'], ids=[b'topdir-id', b'middledir-id', b'file-id'])
    tree.commit('Added files.')
    os.rename('tree/topdir', 'tree/topdir2')
    RenameMap.guess_renames(tree.basis_tree(), tree)
    self.assertEqual('topdir2', tree.id2path(b'topdir-id'))
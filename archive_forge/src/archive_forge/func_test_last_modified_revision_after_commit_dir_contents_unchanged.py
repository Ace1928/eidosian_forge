import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_revision_after_commit_dir_contents_unchanged(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['dir/', 'dir/orig'])
    tree.add(['dir', 'dir/orig'])
    rev1 = tree.commit('rev1')
    self.build_tree(['dir/content'])
    tree.add(['dir/content'])
    rev2 = tree.commit('rev2')
    tree1, tree2 = self._get_revtrees(tree, [rev1, rev2])
    self.assertEqual(rev1, tree1.get_file_revision('dir'))
    self.assertEqual(rev1, tree2.get_file_revision('dir'))
    if tree.supports_file_ids:
        dir_id = tree1.path2id('dir')
        expected_graph = {(dir_id, rev1): ()}
        self.assertFileGraph(expected_graph, tree, (dir_id, rev1))
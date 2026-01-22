import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_revision_tree_record_iter_changes(self):
    tree = self.make_branch_and_tree('.')
    with tree.lock_write():
        builder = tree.branch.get_commit_builder([])
        try:
            list(builder.record_iter_changes(tree, _mod_revision.NULL_REVISION, tree.iter_changes(tree.basis_tree())))
            builder.finish_inventory()
            rev_id = builder.commit('foo bar')
        except:
            builder.abort()
            raise
        rev_tree = builder.revision_tree()
        self.assertEqual(rev_id, rev_tree.get_revision_id())
        self.assertEqual((), tuple(rev_tree.get_parent_ids()))
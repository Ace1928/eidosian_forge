import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_updates_branch(self):
    tree = self.make_branch_and_tree('.')
    with tree.lock_write():
        builder = tree.branch.get_commit_builder([])
        list(builder.record_iter_changes(tree, tree.last_revision(), tree.iter_changes(tree.basis_tree())))
        builder.finish_inventory()
        will_update_branch = builder.updates_branch
        rev_id = builder.commit('might update the branch')
    actually_updated_branch = tree.branch.last_revision() == rev_id
    self.assertEqual(actually_updated_branch, will_update_branch)
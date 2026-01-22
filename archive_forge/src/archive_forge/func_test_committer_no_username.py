import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_committer_no_username(self):
    override_whoami(self)
    tree = self.make_branch_and_tree('.')
    with tree.lock_write():
        self.assertRaises(errors.NoWhoami, tree.branch.get_commit_builder, [])
        builder = tree.branch.get_commit_builder([], committer='me@example.com')
        try:
            list(builder.record_iter_changes(tree, tree.last_revision(), tree.iter_changes(tree.basis_tree())))
            builder.finish_inventory()
        except:
            builder.abort()
            raise
        repo = tree.branch.repository
        repo.commit_write_group()
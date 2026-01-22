import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_revision_after_rename_ref_changes(self):
    tree = self.make_branch_and_tree('.')
    subtree = self.make_reference('reference')
    subtree.commit('')
    try:
        tree.add_reference(subtree)
        self._commit_renamed_check_changed(tree, 'reference')
    except errors.UnsupportedOperation:
        return
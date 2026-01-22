import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_revision_after_merge_new_file_unchanged(self):
    tree1 = self.make_branch_and_tree('t1')
    self._commit_sprout_make_merge(tree1, self.make_file)
import os
from io import BytesIO
from ... import errors
from ... import revision as _mod_revision
from ...bzr.inventory import (Inventory, InventoryDirectory, InventoryFile,
from ...bzr.inventorytree import InventoryRevisionTree, InventoryTree
from ...tests import TestNotApplicable
from ...uncommit import uncommit
from .. import features
from ..per_workingtree import TestCaseWithWorkingTree
def test_add_first_parent_tree(self):
    """Test adding the first parent id"""
    tree = self.make_branch_and_tree('.')
    first_revision = tree.commit('first post')
    uncommit(tree.branch, tree=tree)
    tree.add_parent_tree((first_revision, tree.branch.repository.revision_tree(first_revision)))
    self.assertConsistentParents([first_revision], tree)
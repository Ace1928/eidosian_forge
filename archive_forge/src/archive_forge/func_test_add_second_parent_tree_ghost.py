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
def test_add_second_parent_tree_ghost(self):
    """Test adding the second parent id - as a ghost"""
    tree = self.make_branch_and_tree('.')
    first_revision = tree.commit('first post')
    if tree._format.supports_righthand_parent_id_as_ghost:
        tree.add_parent_tree((b'second', None))
        self.assertConsistentParents([first_revision, b'second'], tree)
    else:
        self.assertRaises(errors.GhostRevisionUnusableHere, tree.add_parent_tree, (b'second', None))
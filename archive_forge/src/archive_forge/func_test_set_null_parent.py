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
def test_set_null_parent(self):
    t = self.make_branch_and_tree('.')
    self.assertRaises(errors.ReservedId, t.set_parent_ids, [b'null:'], allow_leftmost_as_ghost=True)
    self.assertRaises(errors.ReservedId, t.set_parent_trees, [(b'null:', None)], allow_leftmost_as_ghost=True)
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
def test_set_one_ghost_parent_ids_force(self):
    t = self.make_branch_and_tree('.')
    if t._format.supports_leftmost_parent_id_as_ghost:
        t.set_parent_ids([b'missing-revision-id'], allow_leftmost_as_ghost=True)
        self.assertConsistentParents([b'missing-revision-id'], t)
    else:
        self.assertRaises(errors.GhostRevisionUnusableHere, t.set_parent_ids, [b'missing-revision-id'], allow_leftmost_as_ghost=True)
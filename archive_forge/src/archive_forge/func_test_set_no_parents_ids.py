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
def test_set_no_parents_ids(self):
    t = self.make_branch_and_tree('.')
    t.set_parent_ids([])
    self.assertEqual([], t.get_parent_ids())
    t.commit('first post')
    t.set_parent_ids([])
    self.assertConsistentParents([], t)
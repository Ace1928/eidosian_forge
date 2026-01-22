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
def test_set_parent_ids_in_ancestry(self):
    t = self.make_branch_and_tree('.')
    rev1 = t.commit('first post')
    rev2 = t.commit('second post')
    rev3 = t.commit('third post')
    t.set_parent_ids([rev1])
    t.branch.set_last_revision_info(1, rev1)
    self.assertConsistentParents([rev1], t)
    t.set_parent_ids([rev1, rev2, rev3])
    self.assertConsistentParents([rev1, rev3], t)
    t.set_parent_ids([rev2, rev3, rev1])
    self.assertConsistentParents([rev2, rev3], t)
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
def test_set_duplicate_parent_trees(self):
    t = self.make_branch_and_tree('.')
    rev1 = t.commit('first post')
    uncommit(t.branch, tree=t)
    rev2 = t.commit('second post')
    uncommit(t.branch, tree=t)
    rev3 = t.commit('third post')
    uncommit(t.branch, tree=t)
    rev_tree1 = t.branch.repository.revision_tree(rev1)
    rev_tree2 = t.branch.repository.revision_tree(rev2)
    rev_tree3 = t.branch.repository.revision_tree(rev3)
    t.set_parent_trees([(rev1, rev_tree1), (rev2, rev_tree2), (rev2, rev_tree2), (rev3, rev_tree3)])
    self.assertConsistentParents([rev1, rev2, rev3], t)
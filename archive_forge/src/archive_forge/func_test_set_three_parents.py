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
def test_set_three_parents(self):
    t = self.make_branch_and_tree('.')
    first_revision = t.commit('first post')
    uncommit(t.branch, tree=t)
    second_revision = t.commit('second post')
    uncommit(t.branch, tree=t)
    third_revision = t.commit('third post')
    uncommit(t.branch, tree=t)
    rev_tree1 = t.branch.repository.revision_tree(first_revision)
    rev_tree2 = t.branch.repository.revision_tree(second_revision)
    rev_tree3 = t.branch.repository.revision_tree(third_revision)
    t.set_parent_trees([(first_revision, rev_tree1), (second_revision, rev_tree2), (third_revision, rev_tree3)])
    self.assertConsistentParents([first_revision, second_revision, third_revision], t)
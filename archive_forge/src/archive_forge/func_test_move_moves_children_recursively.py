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
def test_move_moves_children_recursively(self):
    old_revid = b'old-parent'
    basis_shape = Inventory(root_id=None)
    self.add_dir(basis_shape, old_revid, b'root-id', None, '')
    self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
    self.add_dir(basis_shape, old_revid, b'dir-id-B', b'dir-id-A', 'B')
    self.add_link(basis_shape, old_revid, b'link-id-C', b'dir-id-B', 'C', 'D')
    new_revid = b'new-parent'
    new_shape = Inventory(root_id=None)
    self.add_new_root(new_shape, old_revid, new_revid)
    self.add_dir(new_shape, new_revid, b'dir-id-A', b'root-id', 'B')
    self.add_dir(new_shape, old_revid, b'dir-id-B', b'dir-id-A', 'B')
    self.add_link(new_shape, old_revid, b'link-id-C', b'dir-id-B', 'C', 'D')
    self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)
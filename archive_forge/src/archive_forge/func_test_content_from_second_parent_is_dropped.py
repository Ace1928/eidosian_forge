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
def test_content_from_second_parent_is_dropped(self):
    left_revid = b'left-parent'
    basis_shape = Inventory(root_id=None)
    self.add_dir(basis_shape, left_revid, b'root-id', None, '')
    self.add_link(basis_shape, left_revid, b'link-id', b'root-id', 'link', 'left-target')
    right_revid = b'right-parent'
    right_shape = Inventory(root_id=None)
    self.add_dir(right_shape, left_revid, b'root-id', None, '')
    self.add_link(right_shape, right_revid, b'link-id', b'root-id', 'link', 'some-target')
    self.add_dir(right_shape, right_revid, b'subdir-id', b'root-id', 'dir')
    self.add_file(right_shape, right_revid, b'file-id', b'subdir-id', 'file', b'2' * 32, 24)
    new_revid = b'new-parent'
    new_shape = Inventory(root_id=None)
    self.add_new_root(new_shape, left_revid, new_revid)
    self.add_link(new_shape, new_revid, b'link-id', b'root-id', 'link', 'new-target')
    self.assertTransitionFromBasisToShape(basis_shape, left_revid, new_shape, new_revid, right_revid)
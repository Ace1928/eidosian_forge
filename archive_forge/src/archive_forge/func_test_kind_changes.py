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
def test_kind_changes(self):

    def do_file(inv, revid):
        self.add_file(inv, revid, b'path-id', b'root-id', 'path', b'1' * 32, 12)

    def do_link(inv, revid):
        self.add_link(inv, revid, b'path-id', b'root-id', 'path', 'target')

    def do_dir(inv, revid):
        self.add_dir(inv, revid, b'path-id', b'root-id', 'path')
    for old_factory in (do_file, do_link, do_dir):
        for new_factory in (do_file, do_link, do_dir):
            if old_factory == new_factory:
                continue
            old_revid = b'old-parent'
            basis_shape = Inventory(root_id=None)
            self.add_dir(basis_shape, old_revid, b'root-id', None, '')
            old_factory(basis_shape, old_revid)
            new_revid = b'new-parent'
            new_shape = Inventory(root_id=None)
            self.add_new_root(new_shape, old_revid, new_revid)
            new_factory(new_shape, new_revid)
            self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)
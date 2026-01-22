import bz2
import os
import sys
import tempfile
from io import BytesIO
from ... import diff, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...tests import features, test_commit
from ...tree import InterTree
from .. import bzrdir, inventory, knitrepo
from ..bundle.apply_bundle import install_bundle, merge_bundle
from ..bundle.bundle_data import BundleTree
from ..bundle.serializer import read_bundle, v4, v09, write_bundle
from ..bundle.serializer.v4 import BundleSerializerV4
from ..bundle.serializer.v08 import BundleSerializerV08
from ..bundle.serializer.v09 import BundleSerializerV09
from ..inventorytree import InventoryTree
def test_binary_bundle(self):
    self.tree1 = self.make_branch_and_tree('b1')
    self.b1 = self.tree1.branch
    tt = self.tree1.transform()
    tt.new_file('file', tt.root, [b'\x00\n\x00\r\x01\n\x02\r\xff'], b'binary-1')
    tt.new_file('file2', tt.root, [b'\x01\n\x02\r\x03\n\x04\r\xff'], b'binary-2')
    tt.apply()
    self.tree1.commit('add binary', rev_id=b'b@cset-0-1')
    self.get_valid_bundle(b'null:', b'b@cset-0-1')
    tt = self.tree1.transform()
    trans_id = tt.trans_id_tree_path('file')
    tt.delete_contents(trans_id)
    tt.apply()
    self.tree1.commit('delete binary', rev_id=b'b@cset-0-2')
    self.get_valid_bundle(b'b@cset-0-1', b'b@cset-0-2')
    tt = self.tree1.transform()
    trans_id = tt.trans_id_tree_path('file2')
    tt.adjust_path('file3', tt.root, trans_id)
    tt.delete_contents(trans_id)
    tt.create_file([b'file\rcontents\x00\n\x00'], trans_id)
    tt.apply()
    self.tree1.commit('rename and modify binary', rev_id=b'b@cset-0-3')
    self.get_valid_bundle(b'b@cset-0-2', b'b@cset-0-3')
    tt = self.tree1.transform()
    trans_id = tt.trans_id_tree_path('file3')
    tt.delete_contents(trans_id)
    tt.create_file([b'\x00file\rcontents'], trans_id)
    tt.apply()
    self.tree1.commit('just modify binary', rev_id=b'b@cset-0-4')
    self.get_valid_bundle(b'b@cset-0-3', b'b@cset-0-4')
    self.get_valid_bundle(b'null:', b'b@cset-0-4')
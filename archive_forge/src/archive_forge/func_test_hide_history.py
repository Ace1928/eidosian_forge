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
def test_hide_history(self):
    self.tree1 = self.make_branch_and_tree('b1')
    self.b1 = self.tree1.branch
    with open('b1/one', 'wb') as f:
        f.write(b'one\n')
    self.tree1.add('one')
    self.tree1.commit('add file', rev_id=b'a@cset-0-1')
    with open('b1/one', 'wb') as f:
        f.write(b'two\n')
    self.tree1.commit('modify', rev_id=b'a@cset-0-2')
    with open('b1/one', 'wb') as f:
        f.write(b'three\n')
    self.tree1.commit('modify', rev_id=b'a@cset-0-3')
    bundle_file = BytesIO()
    rev_ids = write_bundle(self.tree1.branch.repository, b'a@cset-0-3', b'a@cset-0-1', bundle_file, format=self.format)
    self.assertNotContainsRe(bundle_file.getvalue(), b'\x08two\x08')
    self.assertContainsRe(self.get_raw(bundle_file), b'one')
    self.assertContainsRe(self.get_raw(bundle_file), b'three')
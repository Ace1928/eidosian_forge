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
def test_across_serializers(self):
    tree = self.make_simple_tree('knit')
    tree.commit('hello', rev_id=b'rev1')
    tree.commit('hello', rev_id=b'rev2')
    bundle = read_bundle(self.create_bundle_text(b'null:', b'rev2')[0])
    repo = self.make_repository('repo', format='dirstate-with-subtree')
    bundle.install_revisions(repo)
    inv_text = b''.join(repo._get_inventory_xml(b'rev2'))
    self.assertNotContainsRe(inv_text, b'format="5"')
    self.assertContainsRe(inv_text, b'format="7"')
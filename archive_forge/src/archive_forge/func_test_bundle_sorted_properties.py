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
def test_bundle_sorted_properties(self):
    """For stability the writer should write properties in sorted order."""
    tree = self.make_branch_and_memory_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.add([''], ids=[b'TREE_ROOT'])
    tree.commit('One', rev_id=b'rev1', revprops={'a': '4', 'b': '3', 'c': '2', 'd': '1'})
    self.b1 = tree.branch
    bundle_sio, revision_ids = self.create_bundle_text(b'null:', b'rev1')
    self.assertContainsRe(bundle_sio.getvalue(), b'# properties:\n#   a: 4\n#   b: 3\n#   branch-nick: tree\n#   c: 2\n#   d: 1\n')
    bundle = read_bundle(bundle_sio)
    revision_info = bundle.revisions[0]
    self.assertEqual(b'rev1', revision_info.revision_id)
    rev = revision_info.as_revision()
    self.assertEqual({'branch-nick': 'tree', 'a': '4', 'b': '3', 'c': '2', 'd': '1'}, rev.properties)
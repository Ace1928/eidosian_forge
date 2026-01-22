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
def test_alt_timezone_bundle(self):
    self.tree1 = self.make_branch_and_memory_tree('b1')
    self.b1 = self.tree1.branch
    builder = treebuilder.TreeBuilder()
    self.tree1.lock_write()
    builder.start_tree(self.tree1)
    builder.build(['newfile'])
    builder.finish_tree()
    self.tree1.commit('non-hour offset timezone', rev_id=b'tz-1', timezone=19800, timestamp=1152544886.0)
    bundle = self.get_valid_bundle(b'null:', b'tz-1')
    rev = bundle.revisions[0]
    self.assertEqual('Mon 2006-07-10 20:51:26.000000000 +0530', rev.date)
    self.assertEqual(19800, rev.timezone)
    self.assertEqual(1152544886.0, rev.timestamp)
    self.tree1.unlock()
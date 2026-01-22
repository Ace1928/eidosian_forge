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
def test_creating_bundle_preserves_chk_pages(self):
    self.make_merged_branch()
    target = self.b1.controldir.sprout('target', revision_id=b'a@cset-0-2a').open_branch()
    bundle_txt, rev_ids = self.create_bundle_text(b'a@cset-0-2a', b'a@cset-0-3')
    self.assertEqual({b'a@cset-0-2b', b'a@cset-0-3'}, set(rev_ids))
    bundle = read_bundle(bundle_txt)
    target.lock_write()
    self.addCleanup(target.unlock)
    install_bundle(target.repository, bundle)
    inv1 = next(self.b1.repository.inventories.get_record_stream([(b'a@cset-0-3',)], 'unordered', True)).get_bytes_as('fulltext')
    inv2 = next(target.repository.inventories.get_record_stream([(b'a@cset-0-3',)], 'unordered', True)).get_bytes_as('fulltext')
    self.assertEqualDiff(inv1, inv2)
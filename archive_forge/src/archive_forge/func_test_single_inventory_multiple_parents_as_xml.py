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
def test_single_inventory_multiple_parents_as_xml(self):
    self.make_merged_branch()
    sio = self.make_bundle_just_inventories(b'a@cset-0-1', b'a@cset-0-3', [b'a@cset-0-3'])
    reader = v4.BundleReader(sio, stream_input=False)
    records = list(reader.iter_records())
    self.assertEqual(1, len(records))
    bytes, metadata, repo_kind, revision_id, file_id = records[0]
    self.assertIs(None, file_id)
    self.assertEqual(b'a@cset-0-3', revision_id)
    self.assertEqual('inventory', repo_kind)
    self.assertEqual({b'parents': [b'a@cset-0-2a', b'a@cset-0-2b'], b'sha1': b'09c53b0c4de0895e11a2aacc34fef60a6e70865c', b'storage_kind': b'mpdiff'}, metadata)
    self.assertEqualDiff(b'i 1\n<inventory format="10" revision_id="a@cset-0-3">\n\nc 0 1 1 2\nc 1 3 3 2\n', bytes)
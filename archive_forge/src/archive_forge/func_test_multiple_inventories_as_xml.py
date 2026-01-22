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
def test_multiple_inventories_as_xml(self):
    self.make_merged_branch()
    sio = self.make_bundle_just_inventories(b'a@cset-0-1', b'a@cset-0-3', [b'a@cset-0-2a', b'a@cset-0-2b', b'a@cset-0-3'])
    reader = v4.BundleReader(sio, stream_input=False)
    records = list(reader.iter_records())
    self.assertEqual(3, len(records))
    revision_ids = [rev_id for b, m, k, rev_id, f in records]
    self.assertEqual([b'a@cset-0-2a', b'a@cset-0-2b', b'a@cset-0-3'], revision_ids)
    metadata_2a = records[0][1]
    self.assertEqual({b'parents': [b'a@cset-0-1'], b'sha1': b'1e105886d62d510763e22885eec733b66f5f09bf', b'storage_kind': b'mpdiff'}, metadata_2a)
    metadata_2b = records[1][1]
    self.assertEqual({b'parents': [b'a@cset-0-1'], b'sha1': b'f03f12574bdb5ed2204c28636c98a8547544ccd8', b'storage_kind': b'mpdiff'}, metadata_2b)
    metadata_3 = records[2][1]
    self.assertEqual({b'parents': [b'a@cset-0-2a', b'a@cset-0-2b'], b'sha1': b'09c53b0c4de0895e11a2aacc34fef60a6e70865c', b'storage_kind': b'mpdiff'}, metadata_3)
    bytes_2a = records[0][0]
    self.assertEqualDiff(b'i 1\n<inventory format="10" revision_id="a@cset-0-2a">\n\nc 0 1 1 1\ni 1\n<file file_id="file-id" name="file" parent_id="root-id" revision="a@cset-0-2a" text_sha1="50f545ff40e57b6924b1f3174b267ffc4576e9a9" text_size="12" />\n\nc 0 3 3 1\n', bytes_2a)
    bytes_2b = records[1][0]
    self.assertEqualDiff(b'i 1\n<inventory format="10" revision_id="a@cset-0-2b">\n\nc 0 1 1 2\ni 1\n<file file_id="file2-id" name="other-file" parent_id="root-id" revision="a@cset-0-2b" text_sha1="b46c0c8ea1e5ef8e46fc8894bfd4752a88ec939e" text_size="14" />\n\nc 0 3 4 1\n', bytes_2b)
    bytes_3 = records[2][0]
    self.assertEqualDiff(b'i 1\n<inventory format="10" revision_id="a@cset-0-3">\n\nc 0 1 1 2\nc 1 3 3 2\n', bytes_3)
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
def test_decode_name(self):
    self.assertEqual(('revision', b'rev1', None), v4.BundleReader.decode_name(b'revision/rev1'))
    self.assertEqual(('file', b'rev/1', b'file-id-1'), v4.BundleReader.decode_name(b'file/rev//1/file-id-1'))
    self.assertEqual(('info', None, None), v4.BundleReader.decode_name(b'info'))
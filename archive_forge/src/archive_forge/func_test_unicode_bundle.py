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
def test_unicode_bundle(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    os.mkdir('b1')
    f = open('b1/with Dod€', 'wb')
    self.tree1 = self.make_branch_and_tree('b1')
    self.b1 = self.tree1.branch
    f.write('A file\nWith international man of mystery\nWilliam Dodé\n'.encode())
    f.close()
    self.tree1.add(['with Dod€'], ids=[b'withdod-id'])
    self.tree1.commit('i18n commit from William Dodé', rev_id=b'i18n-1', committer='William Dodé')
    bundle = self.get_valid_bundle(b'null:', b'i18n-1')
    f = open('b1/with Dod€', 'wb')
    f.write('Modified µ\n'.encode())
    f.close()
    self.tree1.commit('modified', rev_id=b'i18n-2')
    bundle = self.get_valid_bundle(b'i18n-1', b'i18n-2')
    self.tree1.rename_one('with Dod€', 'B€gfors')
    self.tree1.commit('renamed, the new i18n man', rev_id=b'i18n-3', committer='Erik Bågfors')
    bundle = self.get_valid_bundle(b'i18n-2', b'i18n-3')
    self.tree1.remove(['B€gfors'])
    self.tree1.commit('removed', rev_id=b'i18n-4')
    bundle = self.get_valid_bundle(b'i18n-3', b'i18n-4')
    bundle = self.get_valid_bundle(b'null:', b'i18n-4')
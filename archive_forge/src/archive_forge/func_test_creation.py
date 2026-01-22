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
def test_creation(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/file', b'contents1\nstatic\n')])
    tree.add('file', ids=b'fileid-2')
    tree.commit('added file', rev_id=b'rev1')
    self.build_tree_contents([('tree/file', b'contents2\nstatic\n')])
    tree.commit('changed file', rev_id=b'rev2')
    s = BytesIO()
    serializer = BundleSerializerV4('1.0')
    with tree.lock_read():
        serializer.write_bundle(tree.branch.repository, b'rev2', b'null:', s)
    s.seek(0)
    tree2 = self.make_branch_and_tree('target')
    target_repo = tree2.branch.repository
    install_bundle(target_repo, serializer.read(s))
    target_repo.lock_read()
    self.addCleanup(target_repo.unlock)
    repo_texts = {i: b''.join(content) for i, content in target_repo.iter_files_bytes([(b'fileid-2', b'rev1', '1'), (b'fileid-2', b'rev2', '2')])}
    self.assertEqual({'1': b'contents1\nstatic\n', '2': b'contents2\nstatic\n'}, repo_texts)
    rtree = target_repo.revision_tree(b'rev2')
    inventory_vf = target_repo.inventories
    self.assertSubset([inventory_vf.get_parent_map([(b'rev2',)])[b'rev2',]], [None, ((b'rev1',),)])
    self.assertEqual('changed file', target_repo.get_revision(b'rev2').message)
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
def get_checkout(self, rev_id, checkout_dir=None):
    """Get a new tree, with the specified revision in it.
        """
    if checkout_dir is None:
        checkout_dir = tempfile.mkdtemp(prefix='test-branch-', dir='.')
    elif not os.path.exists(checkout_dir):
        os.mkdir(checkout_dir)
    tree = self.make_branch_and_tree(checkout_dir)
    s = BytesIO()
    ancestors = write_bundle(self.b1.repository, rev_id, b'null:', s, format=self.format)
    s.seek(0)
    self.assertIsInstance(s.getvalue(), bytes)
    install_bundle(tree.branch.repository, read_bundle(s))
    for ancestor in ancestors:
        old = self.b1.repository.revision_tree(ancestor)
        new = tree.branch.repository.revision_tree(ancestor)
        with old.lock_read(), new.lock_read():
            delta = new.changes_from(old)
            self.assertFalse(delta.has_changed(), 'Revision %s not copied correctly.' % (ancestor,))
            for path in old.all_versioned_paths():
                try:
                    old_file = old.get_file(path)
                except _mod_transport.NoSuchFile:
                    continue
                self.assertEqual(old_file.read(), new.get_file(path).read())
    if not _mod_revision.is_null(rev_id):
        tree.branch.generate_revision_history(rev_id)
        tree.update()
        delta = tree.changes_from(self.b1.repository.revision_tree(rev_id))
        self.assertFalse(delta.has_changed(), 'Working tree has modifications: %s' % delta)
    return tree
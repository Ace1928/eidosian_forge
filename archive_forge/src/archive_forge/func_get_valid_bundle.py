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
def get_valid_bundle(self, base_rev_id, rev_id, checkout_dir=None):
    """Create a bundle from base_rev_id -> rev_id in built-in branch.
        Make sure that the text generated is valid, and that it
        can be applied against the base, and generate the same information.

        :return: The in-memory bundle
        """
    bundle_txt, rev_ids = self.create_bundle_text(base_rev_id, rev_id)
    bundle = read_bundle(bundle_txt)
    repository = self.b1.repository
    for bundle_rev in bundle.real_revisions:
        branch_rev = repository.get_revision(bundle_rev.revision_id)
        for a in ('inventory_sha1', 'revision_id', 'parent_ids', 'timestamp', 'timezone', 'message', 'committer', 'parent_ids', 'properties'):
            self.assertEqual(getattr(branch_rev, a), getattr(bundle_rev, a))
        self.assertEqual(len(branch_rev.parent_ids), len(bundle_rev.parent_ids))
    self.assertEqual(set(rev_ids), {r.revision_id for r in bundle.real_revisions})
    self.valid_apply_bundle(base_rev_id, bundle, checkout_dir=checkout_dir)
    return bundle
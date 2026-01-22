import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_unicode_symlink(self):
    inv_entry = inventory.InventoryLink(b'link-file-id', 'nam€e', b'link-parent-id')
    inv_entry.revision = b'link-revision-id'
    target = 'link-targ€t'
    inv_entry.symlink_target = target
    self.assertDetails((b'l', target.encode('UTF-8'), 0, False, b'link-revision-id'), inv_entry)
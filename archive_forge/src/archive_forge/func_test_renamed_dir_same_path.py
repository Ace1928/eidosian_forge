import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_renamed_dir_same_path(self):
    state = self.assertUpdate(active=[('dir/', b'A-id'), ('dir/B', b'B-id')], basis=[('dir/', b'A-id'), ('dir/B', b'B-id')], target=[('dir/', b'C-id'), ('dir/B', b'B-id')])
    state = self.assertUpdate(active=[('dir/', b'C-id'), ('dir/B', b'B-id')], basis=[('dir/', b'A-id'), ('dir/B', b'B-id')], target=[('dir/', b'C-id'), ('dir/B', b'B-id')])
    state = self.assertUpdate(active=[], basis=[('dir/', b'A-id'), ('dir/B', b'B-id')], target=[('dir/', b'C-id'), ('dir/B', b'B-id')])
    state = self.assertUpdate(active=[('dir/', b'D-id'), ('dir/B', b'B-id')], basis=[('dir/', b'A-id'), ('dir/B', b'B-id')], target=[('dir/', b'C-id'), ('dir/B', b'B-id')])
import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_add_file_in_empty_dir_not_matching_active_state(self):
    state = self.assertUpdate(active=[], basis=[('dir/', b'dir-id')], target=[('dir/', b'dir-id', b'basis'), ('dir/file', b'file-id')])
import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_remove_file_active_state_has_diff_file(self):
    state = self.assertUpdate(active=[('file', b'file-id-2')], basis=[('file', b'file-id')], target=[])
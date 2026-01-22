import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_add_file_active_state_has_diff_file_and_file_elsewhere(self):
    state = self.assertUpdate(active=[('other-file', b'file-id'), ('file', b'file-id-2')], basis=[], target=[('file', b'file-id')])
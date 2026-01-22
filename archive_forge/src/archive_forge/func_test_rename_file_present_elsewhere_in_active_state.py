import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_rename_file_present_elsewhere_in_active_state(self):
    state = self.assertUpdate(active=[('third', b'file-id')], basis=[('file', b'file-id')], target=[('other-file', b'file-id')])
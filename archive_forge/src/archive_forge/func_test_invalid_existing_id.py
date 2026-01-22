import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_invalid_existing_id(self):
    state = self.assertBadDelta(active=[('file', b'file-id')], basis=[('file', b'file-id')], delta=[(None, 'file', b'file-id')])
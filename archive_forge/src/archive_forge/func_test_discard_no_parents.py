import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_discard_no_parents(self):
    state = self.create_empty_dirstate()
    self.addCleanup(state.unlock)
    state._discard_merge_parents()
    state._validate()
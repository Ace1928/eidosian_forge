import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_can_save_clean_on_file(self):
    tree = self.make_branch_and_tree('tree')
    state = dirstate.DirState.from_tree(tree, 'dirstate')
    try:
        state.save()
    finally:
        state.unlock()
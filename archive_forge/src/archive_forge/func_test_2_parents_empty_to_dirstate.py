import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_2_parents_empty_to_dirstate(self):
    tree = self.make_branch_and_tree('tree')
    rev_id = tree.commit('first post')
    tree2 = tree.controldir.sprout('tree2').open_workingtree()
    rev_id2 = tree2.commit('second post', allow_pointless=True)
    tree.merge_from_branch(tree2.branch)
    expected_result = ([rev_id, rev_id2], [((b'', b'', tree.path2id('')), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT), (b'd', b'', 0, False, rev_id), (b'd', b'', 0, False, rev_id)])])
    state = dirstate.DirState.from_tree(tree, 'dirstate')
    self.check_state_with_reopen(expected_result, state)
    state.lock_read()
    try:
        state._validate()
    finally:
        state.unlock()
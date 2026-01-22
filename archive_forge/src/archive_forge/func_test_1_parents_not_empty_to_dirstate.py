import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_1_parents_not_empty_to_dirstate(self):
    tree = self.get_tree_with_a_file()
    rev_id = tree.commit('first post')
    self.build_tree_contents([('tree/a file', b'new content\n')])
    expected_result = ([rev_id], [((b'', b'', tree.path2id('')), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT), (b'd', b'', 0, False, rev_id)]), ((b'', b'a file', b'a-file-id'), [(b'f', b'', 0, False, dirstate.DirState.NULLSTAT), (b'f', b'c3ed76e4bfd45ff1763ca206055bca8e9fc28aa8', 24, False, rev_id)])])
    state = dirstate.DirState.from_tree(tree, 'dirstate')
    self.check_state_with_reopen(expected_result, state)
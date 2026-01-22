import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_duplicated_dirstate(self):
    """Create a dirstate with a deleted and added entries.

        This grabs a basic_dirstate, and then removes and re adds every entry
        with a new file id.
        """
    tree, state, expected = self.create_basic_dirstate()
    tree.unversion(['f', 'b-c', 'b/d/e', 'b/d', 'b/c', 'b', 'a'])
    tree.add(['a', 'b', 'b/c', 'b/d', 'b/d/e', 'b-c', 'f'], ids=[b'a-id2', b'b-id2', b'c-id2', b'd-id2', b'e-id2', b'b-c-id2', b'f-id2'])
    for path in [b'a', b'b', b'b/c', b'b/d', b'b/d/e', b'b-c', b'f']:
        orig = expected[path]
        path2 = path + b'2'
        expected[path] = (orig[0], [dirstate.DirState.NULL_PARENT_DETAILS, orig[1][1]])
        new_key = (orig[0][0], orig[0][1], orig[0][2] + b'2')
        expected[path2] = (new_key, [orig[1][0], dirstate.DirState.NULL_PARENT_DETAILS])
    state.unlock()
    try:
        new_state = dirstate.DirState.from_tree(tree, 'dirstate')
        try:
            new_state.save()
        finally:
            new_state.unlock()
    finally:
        state.lock_read()
    return (tree, state, expected)
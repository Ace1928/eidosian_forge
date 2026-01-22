import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_path_id_with_parents(self):
    """Set the root file id in a dirstate with parents"""
    mt = self.make_branch_and_tree('mt')
    mt.set_root_id(b'TREE_ROOT')
    mt.commit('foo', rev_id=b'parent-revid')
    rt = mt.branch.repository.revision_tree(b'parent-revid')
    state = dirstate.DirState.initialize('dirstate')
    state._validate()
    try:
        state.set_parent_trees([(b'parent-revid', rt)], ghosts=[])
        root_entry = ((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, b'x' * 32), (b'd', b'', 0, False, b'parent-revid')])
        self.assertEqual(root_entry, state._get_entry(0, path_utf8=b''))
        self.assertEqual(root_entry, state._get_entry(0, fileid_utf8=b'TREE_ROOT'))
        self.assertEqual((None, None), state._get_entry(0, fileid_utf8=b'Asecond-root-id'))
        state.set_path_id(b'', b'Asecond-root-id')
        state._validate()
        old_root_entry = ((b'', b'', b'TREE_ROOT'), [(b'a', b'', 0, False, b''), (b'd', b'', 0, False, b'parent-revid')])
        new_root_entry = ((b'', b'', b'Asecond-root-id'), [(b'd', b'', 0, False, b''), (b'a', b'', 0, False, b'')])
        expected_rows = [new_root_entry, old_root_entry]
        state._validate()
        self.assertEqual(expected_rows, list(state._iter_entries()))
        self.assertEqual(new_root_entry, state._get_entry(0, path_utf8=b''))
        self.assertEqual(old_root_entry, state._get_entry(1, path_utf8=b''))
        self.assertEqual((None, None), state._get_entry(0, fileid_utf8=b'TREE_ROOT'))
        self.assertEqual(old_root_entry, state._get_entry(1, fileid_utf8=b'TREE_ROOT'))
        self.assertEqual(new_root_entry, state._get_entry(0, fileid_utf8=b'Asecond-root-id'))
        self.assertEqual((None, None), state._get_entry(1, fileid_utf8=b'Asecond-root-id'))
        state.save()
    finally:
        state.unlock()
    state = dirstate.DirState.on_file('dirstate')
    state.lock_read()
    try:
        state._validate()
        self.assertEqual(expected_rows, list(state._iter_entries()))
    finally:
        state.unlock()
    state.lock_write()
    try:
        state._validate()
        state.set_path_id(b'', b'tree-root-2')
        state._validate()
    finally:
        state.unlock()
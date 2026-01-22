import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_discard_absent(self):
    """If entries are only in a merge, discard should remove the entries"""
    null_stat = dirstate.DirState.NULLSTAT
    present_dir = (b'd', b'', 0, False, null_stat)
    present_file = (b'f', b'', 0, False, null_stat)
    absent = dirstate.DirState.NULL_PARENT_DETAILS
    root_key = (b'', b'', b'a-root-value')
    file_in_root_key = (b'', b'file-in-root', b'a-file-id')
    file_in_merged_key = (b'', b'file-in-merged', b'b-file-id')
    dirblocks = [(b'', [(root_key, [present_dir, present_dir, present_dir])]), (b'', [(file_in_merged_key, [absent, absent, present_file]), (file_in_root_key, [present_file, present_file, present_file])])]
    state = self.create_empty_dirstate()
    self.addCleanup(state.unlock)
    state._set_data([b'parent-id', b'merged-id'], dirblocks[:])
    state._validate()
    exp_dirblocks = [(b'', [(root_key, [present_dir, present_dir])]), (b'', [(file_in_root_key, [present_file, present_file])])]
    state._discard_merge_parents()
    state._validate()
    self.assertEqual(exp_dirblocks, state._dirblocks)
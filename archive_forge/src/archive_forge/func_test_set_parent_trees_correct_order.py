import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_parent_trees_correct_order(self):
    """After calling set_parent_trees() we should maintain the order."""
    dirs = ['a', 'a-a', 'a/a']
    null_sha = b''
    state = dirstate.DirState.initialize('dirstate')
    self.addCleanup(state.unlock)
    fake_stat = os.stat('dirstate')
    for d in dirs:
        d_id = d.encode('utf-8').replace(b'/', b'_') + b'-id'
        file_path = d + '/f'
        file_id = file_path.encode('utf-8').replace(b'/', b'_') + b'-id'
        state.add(d, d_id, 'directory', fake_stat, null_sha)
        state.add(file_path, file_id, 'file', fake_stat, null_sha)
    expected = [b'', b'', b'a', b'a/a', b'a-a']
    dirblock_names = [d[0] for d in state._dirblocks]
    self.assertEqual(expected, dirblock_names)
    repo = self.make_repository('repo')
    empty_tree = repo.revision_tree(_mod_revision.NULL_REVISION)
    state.set_parent_trees([('null:', empty_tree)], [])
    dirblock_names = [d[0] for d in state._dirblocks]
    self.assertEqual(expected, dirblock_names)
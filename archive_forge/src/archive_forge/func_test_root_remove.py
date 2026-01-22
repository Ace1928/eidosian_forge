import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def test_root_remove(self):
    tree = self.make_branch_and_tree('.', format=self._format)
    self.build_tree_contents([('afile', 'somecontents')])
    tree.add(['afile'])
    tree.commit('Afile')
    with Workspace(tree, use_inotify=self._use_inotify) as ws:
        os.remove('afile')
        changes = list(ws.iter_changes())
        self.assertEqual(1, len(changes), changes)
        self.assertEqual(('afile', None), changes[0].path)
        ws.commit(message='Commit message')
        self.assertEqual(list(ws.iter_changes()), [])
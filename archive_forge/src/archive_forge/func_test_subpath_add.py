import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def test_subpath_add(self):
    tree = self.make_branch_and_tree('.', format=self._format)
    self.build_tree(['subpath/'])
    tree.add('subpath')
    tree.commit('add subpath')
    with Workspace(tree, subpath='subpath', use_inotify=self._use_inotify) as ws:
        self.build_tree_contents([('outside', 'somecontents')])
        self.build_tree_contents([('subpath/afile', 'somecontents')])
        changes = [c for c in ws.iter_changes() if c.path[1] != 'subpath']
        self.assertEqual(1, len(changes), changes)
        self.assertEqual((None, 'subpath/afile'), changes[0].path)
        ws.commit(message='Commit message')
        self.assertEqual(list(ws.iter_changes()), [])
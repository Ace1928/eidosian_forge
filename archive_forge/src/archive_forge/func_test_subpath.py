import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def test_subpath(self):
    tree = self.make_test_tree()
    self.build_tree_contents([('debian/foo', 'blah'), ('foo/',)])
    tree.add('foo')
    tree.commit('add foo')
    with tree.lock_write():
        check_clean_tree(tree, tree.basis_tree(), subpath='foo')
        self.assertRaises(WorkspaceDirty, check_clean_tree, tree, tree.basis_tree(), subpath='')
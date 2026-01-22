import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_directory(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'a/c/', 'a/c/d', 'e/'])
    tree.add(['a', 'a/b', 'a/c', 'a/c/d', 'e'])
    tree.commit('initial')
    tree.rename_one('a', 'e/f')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('e/', 'e/'), ('e/f/', 'a/'), ('e/f/b', 'a/b'), ('e/f/c/', 'a/c/'), ('e/f/c/d', 'a/c/d')])
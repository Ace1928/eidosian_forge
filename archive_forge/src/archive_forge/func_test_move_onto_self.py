import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_onto_self(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['b/', 'b/a'])
    tree.add(['b', 'b/a'])
    tree.commit('initial')
    self.assertRaises(errors.BzrMoveFailedError, tree.move, ['b/a'], 'b')
    tree._validate()
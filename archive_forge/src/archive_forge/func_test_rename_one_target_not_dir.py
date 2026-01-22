import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_target_not_dir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    tree.add(['a'])
    tree.commit('initial')
    self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', 'not-a-dir/b')
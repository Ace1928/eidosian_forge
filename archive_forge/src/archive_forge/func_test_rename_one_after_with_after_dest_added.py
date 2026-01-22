import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_after_with_after_dest_added(self):
    """ using after with a newly added file should work """
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    tree.add(['a'])
    tree.commit('initial')
    os.rename('a', 'b')
    tree.add(['b'])
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b', None)])
    tree.rename_one('a', 'b', after=True)
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b', 'a')])
import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_after_source_removed(self):
    """Rename even if the source was already unversioned."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/'])
    tree.add(['a', 'b'])
    tree.commit('initial')
    os.rename('a', 'b/foo')
    tree.remove(['a'])
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/')])
    tree.rename_one('a', 'b/foo')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/'), ('b/foo', 'a')])
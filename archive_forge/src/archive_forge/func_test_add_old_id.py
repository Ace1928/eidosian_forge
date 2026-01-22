from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_old_id(self):
    """We can add an old id, as long as it doesn't exist now."""
    tree = self.make_branch_and_tree('.')
    if not tree.supports_setting_file_ids():
        self.skipTest('tree does not support setting file ids')
    self.build_tree(['a', 'b'])
    tree.add(['a'])
    file_id = tree.path2id('a')
    tree.commit('first')
    tree.unversion(['a'])
    tree.add(['b'], ids=[file_id])
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b', 'a')])
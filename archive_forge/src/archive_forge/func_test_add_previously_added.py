from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_previously_added(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    tree.add(['foo'])
    tree.unversion(['foo'])
    tree.add(['foo'])
    self.assertTrue(tree.has_filename('foo'))
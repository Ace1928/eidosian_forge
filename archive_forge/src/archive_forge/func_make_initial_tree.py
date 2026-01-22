from breezy import workingtree
from breezy.tests import TestCaseWithTransport
def make_initial_tree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/foo', 'tree/dir/', 'tree/dir/bar'])
    tree.add(['foo', 'dir', 'dir/bar'])
    tree.commit('first')
    return tree
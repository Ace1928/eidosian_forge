from breezy.tests.per_tree import TestCaseWithTree
def get_simple_tree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/one', b'first\ncontent\n')])
    tree.add(['one'])
    rev_1 = tree.commit('one')
    self.build_tree_contents([('tree/one', b'second\ncontent\n')])
    rev_2 = tree.commit('two')
    return (self._convert_tree(tree), [rev_1, rev_2])
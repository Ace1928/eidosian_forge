from breezy import branch
from breezy.tests import TestNotApplicable
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.transport import NoSuchFile
def test_new_path(self):
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    self.build_tree_contents([('1/file', b'apples')])
    tree1.add('file')
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    inter = self.intertree_class(tree1, tree2)
    self.assertIs(None, inter.find_target_path('file'))
    self.assertEqual({'file': None}, inter.find_target_paths(['file']))
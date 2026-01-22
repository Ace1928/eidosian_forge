from breezy import branch
from breezy.tests import TestNotApplicable
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.transport import NoSuchFile
def test_old_path(self):
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    self.build_tree_contents([('2/file', b'apples')])
    tree2.add('file')
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    inter = self.intertree_class(tree1, tree2)
    self.assertIs(None, inter.find_source_path('file'))
    self.assertEqual({'file': None}, inter.find_source_paths(['file']))
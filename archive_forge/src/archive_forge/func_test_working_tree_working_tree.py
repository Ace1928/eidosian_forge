from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
def test_working_tree_working_tree(self):
    tree = self.make_branch_and_tree('1')
    tree2 = self.make_branch_and_tree('2')
    optimiser = InterTree.get(tree, tree2)
    self.assertIsInstance(optimiser, InterTree)
    optimiser = InterTree.get(tree2, tree)
    self.assertIsInstance(optimiser, InterTree)
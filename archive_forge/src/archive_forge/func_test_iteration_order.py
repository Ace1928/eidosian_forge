from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_iteration_order(self):
    work_tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b/', 'a/b/c', 'a/d/', 'a/d/e', 'f/', 'f/g'])
    work_tree.add(['a', 'a/b', 'a/b/c', 'a/d', 'a/d/e', 'f', 'f/g'])
    tree = self._convert_tree(work_tree)
    output = [e.name for e in tree.iter_child_entries('')]
    self.assertEqual({'a', 'f'}, set(output))
    output = [e.name for e in tree.iter_child_entries('a')]
    self.assertEqual({'b', 'd'}, set(output))
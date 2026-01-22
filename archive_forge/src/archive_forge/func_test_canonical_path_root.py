from breezy import tests
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_canonical_path_root(self):
    work_tree = self._make_canonical_test_tree()
    self.assertEqual('', work_tree.get_canonical_path(''))
    self.assertEqual('', work_tree.get_canonical_path('/'))
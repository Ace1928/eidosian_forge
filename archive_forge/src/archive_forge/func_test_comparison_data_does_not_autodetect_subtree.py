from breezy.bzr import inventory, inventorytree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_comparison_data_does_not_autodetect_subtree(self):
    tree = self.prepare_with_subtree()
    path, versioned, kind, ie = list(tree.list_files('subtree'))[0]
    self.assertEqual('directory', tree._comparison_data(ie, 'subtree')[0])
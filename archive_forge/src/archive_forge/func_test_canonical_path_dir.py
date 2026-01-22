from breezy import tests
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_canonical_path_dir(self):
    work_tree = self._make_canonical_test_tree()
    if features.CaseInsensitiveFilesystemFeature.available():
        self.assertEqual('dir', work_tree.get_canonical_path('Dir'))
    elif features.CaseInsCasePresFilenameFeature.available():
        self.assertEqual('dir', work_tree.get_canonical_path('Dir'))
    else:
        self.assertEqual('Dir', work_tree.get_canonical_path('Dir'))
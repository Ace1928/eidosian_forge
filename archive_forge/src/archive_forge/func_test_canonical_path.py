from breezy import tests
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_canonical_path(self):
    work_tree = self._make_canonical_test_tree()
    if features.CaseInsensitiveFilesystemFeature.available():
        self.assertEqual('dir/file', work_tree.get_canonical_path('Dir/File'))
    elif features.CaseInsCasePresFilenameFeature.available():
        self.assertEqual('dir/file', work_tree.get_canonical_path('Dir/File'))
    else:
        self.assertEqual('Dir/File', work_tree.get_canonical_path('Dir/File'))
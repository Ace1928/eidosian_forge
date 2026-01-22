import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
from breezy.tests.features import SymlinkFeature
from breezy.transform import PreviewTree
def test_symlink_content_summary(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    tree = self.make_branch_and_tree('tree')
    os.symlink('target', 'tree/path')
    tree.add(['path'])
    summary = self._convert_tree(tree).path_content_summary('path')
    self.assertEqual(('symlink', None, None, 'target'), summary)
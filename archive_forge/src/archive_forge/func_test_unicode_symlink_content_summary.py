import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
from breezy.tests.features import SymlinkFeature
from breezy.transform import PreviewTree
def test_unicode_symlink_content_summary(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('tree')
    os.symlink('target', os.fsencode('tree/β-path'))
    tree.add(['β-path'])
    summary = self._convert_tree(tree).path_content_summary('β-path')
    self.assertEqual(('symlink', None, None, 'target'), summary)
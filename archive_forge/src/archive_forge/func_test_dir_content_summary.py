import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
from breezy.tests.features import SymlinkFeature
from breezy.transform import PreviewTree
def test_dir_content_summary(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/path/'])
    tree.add(['path'])
    converted_tree = self._convert_tree(tree)
    summary = converted_tree.path_content_summary('path')
    if converted_tree.has_versioned_directories() or converted_tree.has_filename('path'):
        self.assertEqual(('directory', None, None, None), summary)
    else:
        self.assertEqual(('missing', None, None, None), summary)
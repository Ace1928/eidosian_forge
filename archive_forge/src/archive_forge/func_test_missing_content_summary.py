import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
from breezy.tests.features import SymlinkFeature
from breezy.transform import PreviewTree
def test_missing_content_summary(self):
    tree = self.make_branch_and_tree('tree')
    summary = self._convert_tree(tree).path_content_summary('path')
    self.assertEqual(('missing', None, None, None), summary)
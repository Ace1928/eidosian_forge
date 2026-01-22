import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
from breezy.tests.features import SymlinkFeature
from breezy.transform import PreviewTree
def test_tree_content_summary(self):
    tree = self.make_branch_and_tree('tree')
    if not tree.branch.repository._format.supports_tree_reference:
        raise tests.TestNotApplicable('Tree references not supported.')
    subtree = self.make_branch_and_tree('tree/path')
    subtree.commit('')
    tree.add(['path'])
    summary = self._convert_tree(tree).path_content_summary('path')
    self.assertEqual(4, len(summary))
    self.assertEqual('tree-reference', summary[0])
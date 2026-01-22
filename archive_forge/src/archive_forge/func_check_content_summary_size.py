import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
from breezy.tests.features import SymlinkFeature
from breezy.transform import PreviewTree
def check_content_summary_size(self, tree, summary, expected_size):
    returned_size = summary[1]
    if returned_size == expected_size or (tree.supports_content_filtering() and returned_size is None):
        pass
    else:
        self.fail('invalid size in summary: {!r}'.format(returned_size))
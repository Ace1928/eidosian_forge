import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
from breezy.tests.features import SymlinkFeature
from breezy.transform import PreviewTree
def test_file_content_summary_non_exec(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/path'])
    tree.add(['path'])
    summary = self._convert_tree(tree).path_content_summary('path')
    self.assertEqual(4, len(summary))
    self.assertEqual('file', summary[0])
    self.check_content_summary_size(tree, summary, 22)
    self.assertEqual(False, summary[2])
    self.assertSubset((summary[3],), (None, b'0c352290ae1c26ca7f97d5b2906c4624784abd60'))
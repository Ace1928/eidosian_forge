import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_label_modified(self):
    super().make_example_branch()
    self.build_tree_contents([('hello', b'barbar')])
    diff = self.run_bzr('diff', retcode=1)
    self.assertTrue("=== modified file 'hello'" in diff[0])
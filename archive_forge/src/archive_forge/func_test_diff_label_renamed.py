import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_label_renamed(self):
    tree = super().make_example_branch()
    tree.rename_one('hello', 'gruezi')
    diff = self.run_bzr('diff', retcode=1)
    self.assertTrue("=== renamed file 'hello' => 'gruezi'" in diff[0])
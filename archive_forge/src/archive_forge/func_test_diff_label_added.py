import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_label_added(self):
    tree = super().make_example_branch()
    self.build_tree_contents([('barbar', b'barbar')])
    tree.add('barbar')
    diff = self.run_bzr('diff', retcode=1)
    self.assertTrue("=== added file 'barbar'" in diff[0])
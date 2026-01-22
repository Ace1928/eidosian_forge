import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_prefix(self):
    """diff --prefix appends to filenames in output"""
    self.make_example_branch()
    self.build_tree_contents([('hello', b'hello world!\n')])
    out, err = self.run_bzr('diff --prefix old/:new/', retcode=1)
    self.assertEqual(err, '')
    self.assertEqualDiff(subst_dates(out), "=== modified file 'hello'\n--- old/hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ new/hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -1,1 +1,1 @@\n-foo\n+hello world!\n\n")
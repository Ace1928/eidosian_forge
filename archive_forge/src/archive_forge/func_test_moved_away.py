import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_moved_away(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('a', 'asdf\n')])
    tree.add(['a'])
    tree.commit('add a')
    tree.rename_one('a', 'b')
    self.build_tree_contents([('a', 'qwer\n')])
    tree.add('a')
    output, error = self.run_bzr('diff -p0', retcode=1)
    self.assertEqualDiff("=== added file 'a'\n--- a\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ a\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -0,0 +1,1 @@\n+qwer\n\n=== renamed file 'a' => 'b'\n", subst_dates(output))
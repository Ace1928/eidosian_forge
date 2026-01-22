import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_pull_preview(self):
    self.pullable_branch()
    out, err = self.run_bzr('merge --pull --preview -d a b')
    self.assertThat(out, matchers.DocTestMatches("=== modified file 'file'\n--- file\t...\n+++ file\t...\n@@ -1,1 +1,1 @@\n-bar\n+foo\n\n", doctest.ELLIPSIS | doctest.REPORT_UDIFF))
    tree_a = workingtree.WorkingTree.open('a')
    self.assertEqual([self.id1], tree_a.get_parent_ids())
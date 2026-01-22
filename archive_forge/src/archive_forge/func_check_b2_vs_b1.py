import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def check_b2_vs_b1(self, cmd):
    out, err = self.run_bzr(cmd, retcode=1)
    self.assertEqual('', err)
    self.assertEqual("=== modified file 'file'\n--- old/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ new/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -1,1 +1,1 @@\n-new content\n+contents of branch1/file\n\n", subst_dates(out))
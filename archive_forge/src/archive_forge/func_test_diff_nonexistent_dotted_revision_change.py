import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_nonexistent_dotted_revision_change(self):
    out, err = self.run_bzr('diff -c 1.1', retcode=3)
    self.assertContainsRe(err, "Requested revision: '1.1' does not exist in branch:")
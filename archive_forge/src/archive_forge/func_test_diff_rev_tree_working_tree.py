import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
def test_diff_rev_tree_working_tree(self):
    output = get_diff_as_string(self.wt.basis_tree(), self.wt)
    self.assertEqualDiff(output, b"=== modified file 'file1'\n--- old/file1\t2006-04-02 00:00:00 +0000\n+++ new/file1\t2006-04-05 00:00:00 +0000\n@@ -1,1 +1,1 @@\n-file1 contents at rev 2\n+file1 contents in working tree\n\n")
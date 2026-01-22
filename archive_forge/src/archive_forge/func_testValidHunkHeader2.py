import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testValidHunkHeader2(self):
    """Parse a tricky, valid hunk header"""
    header = b'@@ -1 +0,0 @@\n'
    hunk = hunk_from_header(header)
    self.assertEqual(hunk.orig_pos, 1)
    self.assertEqual(hunk.orig_range, 1)
    self.assertEqual(hunk.mod_pos, 0)
    self.assertEqual(hunk.mod_range, 0)
    self.assertEqual(hunk.as_bytes(), header)
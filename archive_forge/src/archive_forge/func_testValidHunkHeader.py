import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testValidHunkHeader(self):
    """Parse a valid hunk header"""
    header = b'@@ -34,11 +50,6 @@\n'
    hunk = hunk_from_header(header)
    self.assertEqual(hunk.orig_pos, 34)
    self.assertEqual(hunk.orig_range, 11)
    self.assertEqual(hunk.mod_pos, 50)
    self.assertEqual(hunk.mod_range, 6)
    self.assertEqual(hunk.as_bytes(), header)
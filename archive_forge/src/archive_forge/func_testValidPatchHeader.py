import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testValidPatchHeader(self):
    """Parse a valid patch header"""
    lines = b'--- orig/commands.py\t2020-09-09 23:39:35 +0000\n+++ mod/dommands.py\t2020-09-09 23:39:35 +0000\n'.split(b'\n')
    orig, mod = get_patch_names(lines.__iter__())
    self.assertEqual(orig, (b'orig/commands.py', b'2020-09-09 23:39:35 +0000'))
    self.assertEqual(mod, (b'mod/dommands.py', b'2020-09-09 23:39:35 +0000'))
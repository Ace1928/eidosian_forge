import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testValidPatchHeaderMissingTimestamps(self):
    """Parse a valid patch header"""
    lines = b'--- orig/commands.py\n+++ mod/dommands.py\n'.split(b'\n')
    orig, mod = get_patch_names(lines.__iter__())
    self.assertEqual(orig, (b'orig/commands.py', None))
    self.assertEqual(mod, (b'mod/dommands.py', None))
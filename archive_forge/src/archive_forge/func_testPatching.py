import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testPatching(self):
    """Test a few patch files, and make sure they work."""
    files = [('diff-2', 'orig-2', 'mod-2'), ('diff-3', 'orig-3', 'mod-3'), ('diff-4', 'orig-4', 'mod-4'), ('diff-5', 'orig-5', 'mod-5'), ('diff-6', 'orig-6', 'mod-6'), ('diff-7', 'orig-7', 'mod-7')]
    for diff, orig, mod in files:
        patch = self.datafile(diff)
        orig_lines = list(self.datafile(orig))
        mod_lines = list(self.datafile(mod))
        patched_file = IterableFile(iter_patched(orig_lines, patch))
        count = 0
        for patch_line in patched_file:
            self.assertEqual(patch_line, mod_lines[count])
            count += 1
        self.assertEqual(count, len(mod_lines))
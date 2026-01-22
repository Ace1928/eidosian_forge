import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def compare_parsed(self, patchtext):
    lines = patchtext.splitlines(True)
    patch = parse_patch(lines.__iter__())
    pstr = patch.as_bytes()
    i = difference_index(patchtext, pstr)
    if i is not None:
        print('%i: "%s" != "%s"' % (i, patchtext[i], pstr[i]))
    self.assertEqual(patchtext, patch.as_bytes())
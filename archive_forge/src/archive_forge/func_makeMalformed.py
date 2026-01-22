import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def makeMalformed(self, header):
    self.assertRaises(MalformedHunkHeader, hunk_from_header, header)
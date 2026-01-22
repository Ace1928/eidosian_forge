import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def test_iter_patched_binary(self):
    binary_lines = self.data_lines('binary.patch')
    e = self.assertRaises(BinaryFiles, iter_patched, [], binary_lines)
import os
from breezy.errors import BinaryFile
from breezy.iterablefile import IterableFile
from breezy.patch import (PatchInvokeError, diff3, iter_patched_from_hunks,
from breezy.patches import parse_patch
from breezy.tests import TestCase, TestCaseInTempDir
def test_missing_patch(self):
    self.assertRaises(PatchInvokeError, run_patch, '.', [], _patch_cmd='/unlikely/to/exist')
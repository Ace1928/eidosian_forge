import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def lineThing(self, text, type):
    line = parse_line(text)
    self.assertIsInstance(line, type)
    self.assertEqual(line.as_bytes(), text)
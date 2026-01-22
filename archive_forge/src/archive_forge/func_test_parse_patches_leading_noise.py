import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def test_parse_patches_leading_noise(self):
    lines = [b'diff -pruN commands.py', b'--- orig/commands.py', b'+++ mod/dommands.py']
    bits = list(parse_patches(iter(lines), allow_dirty=True))
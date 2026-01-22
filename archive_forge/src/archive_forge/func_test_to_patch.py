from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_to_patch(self):
    self.assertEqual([b'c 0 1 2 3\n'], list(multiparent.ParentText(0, 1, 2, 3).to_patch()))
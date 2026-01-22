from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_compare_no_parent(self):
    diff = multiparent.MultiParent.from_lines(LINES_1)
    self.assertEqual([multiparent.NewText(LINES_1)], diff.hunks)
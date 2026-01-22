from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_compare_two_parents(self):
    diff = multiparent.MultiParent.from_lines(LINES_1, [LINES_2, LINES_3])
    self.assertEqual([multiparent.ParentText(1, 0, 0, 4), multiparent.ParentText(0, 3, 4, 1)], diff.hunks)
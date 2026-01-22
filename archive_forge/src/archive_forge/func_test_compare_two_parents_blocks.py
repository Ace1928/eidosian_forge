from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_compare_two_parents_blocks(self):
    matcher = patiencediff.PatienceSequenceMatcher(None, LINES_2, LINES_1)
    blocks = matcher.get_matching_blocks()
    diff = multiparent.MultiParent.from_lines(LINES_1, [LINES_2, LINES_3], left_blocks=blocks)
    self.assertEqual([multiparent.ParentText(1, 0, 0, 4), multiparent.ParentText(0, 3, 4, 1)], diff.hunks)
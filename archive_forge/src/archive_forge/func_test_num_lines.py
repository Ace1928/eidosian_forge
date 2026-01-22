from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_num_lines(self):
    mp = multiparent.MultiParent([multiparent.NewText([b'a\n'])])
    self.assertEqual(1, mp.num_lines())
    mp.hunks.append(multiparent.NewText([b'b\n', b'c\n']))
    self.assertEqual(3, mp.num_lines())
    mp.hunks.append(multiparent.ParentText(0, 0, 3, 2))
    self.assertEqual(5, mp.num_lines())
    mp.hunks.append(multiparent.NewText([b'f\n', b'g\n']))
    self.assertEqual(7, mp.num_lines())
from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_from_patch(self):
    self.assertEqual(multiparent.MultiParent([multiparent.NewText([b'a\n']), multiparent.ParentText(0, 1, 2, 3)]), multiparent.MultiParent.from_patch(b'i 1\na\n\nc 0 1 2 3'))
    self.assertEqual(multiparent.MultiParent([multiparent.NewText([b'a']), multiparent.ParentText(0, 1, 2, 3)]), multiparent.MultiParent.from_patch(b'i 1\na\nc 0 1 2 3\n'))
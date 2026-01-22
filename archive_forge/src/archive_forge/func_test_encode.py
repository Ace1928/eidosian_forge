import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_encode(self):
    self.assertEqual(encode(3, 'abdcd', LinePart(1, 4, 'bdc')), 'a<bd|c>d')
    self.assertEqual(encode(1, 'abdcd', LinePart(1, 4, 'bdc')), 'a|<bdc>d')
    self.assertEqual(encode(4, 'abdcd', LinePart(1, 4, 'bdc')), 'a<bdc|>d')
    self.assertEqual(encode(5, 'abdcd', LinePart(1, 4, 'bdc')), 'a<bdc>d|')
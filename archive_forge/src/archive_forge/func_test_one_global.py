from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import unittest
from pasta.base import test_utils
def test_one_global(self):
    src = 'X = 1\n'
    t = ast.parse(src)
    self.checkAstsEqual(t, t)
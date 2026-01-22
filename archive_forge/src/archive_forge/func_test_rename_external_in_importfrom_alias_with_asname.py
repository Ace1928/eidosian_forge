from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import unittest
from pasta.augment import rename
from pasta.base import scope
from pasta.base import test_utils
def test_rename_external_in_importfrom_alias_with_asname(self):
    src = 'from aaa.bbb import ccc as abc\nabc.foo()'
    t = ast.parse(src)
    self.assertTrue(rename.rename_external(t, 'aaa.bbb.ccc', 'xxx.yyy'))
    self.checkAstsEqual(t, ast.parse('from xxx import yyy as abc\nabc.foo()'))
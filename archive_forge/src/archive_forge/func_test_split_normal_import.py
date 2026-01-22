from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import traceback
import unittest
import pasta
from pasta.augment import import_utils
from pasta.base import ast_utils
from pasta.base import test_utils
from pasta.base import scope
import a
import b
from my_module import a
import b
from my_module import c
from my_module import a, b
from my_module import a as a_mod, b as unused_b_mod
import c as c_mod, d as unused_d_mod
import a
import b
import c
import b
import d
import a, b
import b, c
import d, a, e, f
import a, b, c
import b, c
import a.b
from a import b
import a
import a as ax
import a as ax2
import a as ax
def test_split_normal_import(self):
    src = 'import aaa, bbb, ccc\n'
    t = ast.parse(src)
    import_node = t.body[0]
    sc = scope.analyze(t)
    import_utils.split_import(sc, import_node, import_node.names[1])
    self.assertEqual(2, len(t.body))
    self.assertEqual(ast.Import, type(t.body[1]))
    self.assertEqual([alias.name for alias in t.body[0].names], ['aaa', 'ccc'])
    self.assertEqual([alias.name for alias in t.body[1].names], ['bbb'])
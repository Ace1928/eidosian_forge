from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pasta
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import test_utils
def testReplaceChildInBody(self):
    src = 'def foo():\n  a = 0\n  a += 1 # replace this\n  return a\n'
    replace_with = pasta.parse('foo(a + 1)  # trailing comment\n').body[0]
    expected = 'def foo():\n  a = 0\n  foo(a + 1) # replace this\n  return a\n'
    t = pasta.parse(src)
    parent = t.body[0]
    node_to_replace = parent.body[1]
    ast_utils.replace_child(parent, node_to_replace, replace_with)
    self.assertEqual(expected, pasta.dump(t))
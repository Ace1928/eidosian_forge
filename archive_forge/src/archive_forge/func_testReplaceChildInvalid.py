from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pasta
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import test_utils
def testReplaceChildInvalid(self):
    src = 'def foo():\n  return 1\nx = 1\n'
    replace_with = pasta.parse('bar()').body[0]
    t = pasta.parse(src)
    parent = t.body[0]
    node_to_replace = t.body[1]
    with self.assertRaises(errors.InvalidAstError):
        ast_utils.replace_child(parent, node_to_replace, replace_with)
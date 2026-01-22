from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pasta
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import test_utils
def testRemoveAlias(self):
    src = 'from a import b, c'
    tree = pasta.parse(src)
    import_node = tree.body[0]
    alias1 = import_node.names[0]
    ast_utils.remove_child(import_node, alias1)
    self.assertEqual(pasta.dump(tree), 'from a import c')
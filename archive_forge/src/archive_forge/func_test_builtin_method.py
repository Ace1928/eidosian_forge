from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Symtab import ModuleScope
from Cython.TestUtils import TransformTest
from Cython.Compiler.Visitor import MethodDispatcherTransform
from Cython.Compiler.ParseTreeTransforms import (
def test_builtin_method(self):
    calls = [0]

    class Test(MethodDispatcherTransform):

        def _handle_simple_method_dict_get(self, node, func, args, unbound):
            calls[0] += 1
            return node
    tree = self._build_tree()
    Test(None)(tree)
    self.assertEqual(1, calls[0])
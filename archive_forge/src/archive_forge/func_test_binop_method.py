from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Symtab import ModuleScope
from Cython.TestUtils import TransformTest
from Cython.Compiler.Visitor import MethodDispatcherTransform
from Cython.Compiler.ParseTreeTransforms import (
def test_binop_method(self):
    calls = {'bytes': 0, 'object': 0}

    class Test(MethodDispatcherTransform):

        def _handle_simple_method_bytes___mul__(self, node, func, args, unbound):
            calls['bytes'] += 1
            return node

        def _handle_simple_method_object___mul__(self, node, func, args, unbound):
            calls['object'] += 1
            return node
    tree = self._build_tree()
    Test(None)(tree)
    self.assertEqual(1, calls['bytes'])
    self.assertEqual(0, calls['object'])
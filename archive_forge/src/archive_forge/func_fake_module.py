from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Symtab import ModuleScope
from Cython.TestUtils import TransformTest
from Cython.Compiler.Visitor import MethodDispatcherTransform
from Cython.Compiler.ParseTreeTransforms import (
def fake_module(node):
    scope = ModuleScope('test', None, None)
    return ModuleNode(node.pos, doc=None, body=node, scope=scope, full_module_name='test', directive_comments={})
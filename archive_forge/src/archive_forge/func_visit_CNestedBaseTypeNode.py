from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_CNestedBaseTypeNode(self, node):
    self.visit(node.base_type)
    self.put(u'.')
    self.put(node.name)
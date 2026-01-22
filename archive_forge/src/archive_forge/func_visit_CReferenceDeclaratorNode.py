from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_CReferenceDeclaratorNode(self, node):
    self.put('&')
    self.visit(node.base)
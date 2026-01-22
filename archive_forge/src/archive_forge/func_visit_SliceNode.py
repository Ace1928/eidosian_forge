from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_SliceNode(self, node):
    if not node.start.is_none:
        self.visit(node.start)
    self.put(u':')
    if not node.stop.is_none:
        self.visit(node.stop)
    if not node.step.is_none:
        self.put(u':')
        self.visit(node.step)
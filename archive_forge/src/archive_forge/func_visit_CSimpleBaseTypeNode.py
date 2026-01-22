from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_CSimpleBaseTypeNode(self, node):
    if node.is_basic_c_type:
        self.put(('unsigned ', '', 'signed ')[node.signed])
        if node.longness < 0:
            self.put('short ' * -node.longness)
        elif node.longness > 0:
            self.put('long ' * node.longness)
    if node.name is not None:
        self.put(node.name)
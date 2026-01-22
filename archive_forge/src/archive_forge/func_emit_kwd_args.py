from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def emit_kwd_args(self, node):
    if node is None:
        return
    if isinstance(node, MergedDictNode):
        for expr in node.subexpr_nodes():
            self.emit_kwd_args(expr)
    elif isinstance(node, DictNode):
        for expr in node.subexpr_nodes():
            self.put(u'%s=' % expr.key.value)
            self.visit(expr.value)
            self.put(u', ')
    else:
        self.put(u'**')
        self.visit(node)
        self.put(u', ')
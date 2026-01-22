from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def emit_comprehension(self, body, target, sequence, condition, parens=(u'', u'')):
    open_paren, close_paren = parens
    self.put(open_paren)
    self.visit(body)
    self.put(u' for ')
    self.visit(target)
    self.put(u' in ')
    self.visit(sequence)
    if condition:
        self.put(u' if ')
        self.visit(condition)
    self.put(close_paren)
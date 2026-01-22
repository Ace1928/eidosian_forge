from . import c_ast
def visit_ArrayRef(self, n):
    arrref = self._parenthesize_unless_simple(n.name)
    return arrref + '[' + self.visit(n.subscript) + ']'
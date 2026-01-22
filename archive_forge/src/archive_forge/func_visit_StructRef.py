from . import c_ast
def visit_StructRef(self, n):
    sref = self._parenthesize_unless_simple(n.name)
    return sref + n.type + self.visit(n.field)
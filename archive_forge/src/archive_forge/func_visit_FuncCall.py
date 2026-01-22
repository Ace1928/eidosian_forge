from . import c_ast
def visit_FuncCall(self, n):
    fref = self._parenthesize_unless_simple(n.name)
    return fref + '(' + self.visit(n.args) + ')'
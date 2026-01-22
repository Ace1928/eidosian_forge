from . import c_ast
def visit_Cast(self, n):
    s = '(' + self._generate_type(n.to_type, emit_declname=False) + ')'
    return s + ' ' + self._parenthesize_unless_simple(n.expr)
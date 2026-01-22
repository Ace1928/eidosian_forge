from . import c_ast
def visit_Switch(self, n):
    s = 'switch (' + self.visit(n.cond) + ')\n'
    s += self._generate_stmt(n.stmt, add_indent=True)
    return s
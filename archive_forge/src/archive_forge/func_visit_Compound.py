from . import c_ast
def visit_Compound(self, n):
    s = self._make_indent() + '{\n'
    self.indent_level += 2
    if n.block_items:
        s += ''.join((self._generate_stmt(stmt) for stmt in n.block_items))
    self.indent_level -= 2
    s += self._make_indent() + '}\n'
    return s
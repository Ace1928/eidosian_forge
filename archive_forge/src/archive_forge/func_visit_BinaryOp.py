from . import c_ast
def visit_BinaryOp(self, n):
    lval_str = self._parenthesize_if(n.left, lambda d: not (self._is_simple_node(d) or (self.reduce_parentheses and isinstance(d, c_ast.BinaryOp) and (self.precedence_map[d.op] >= self.precedence_map[n.op]))))
    rval_str = self._parenthesize_if(n.right, lambda d: not (self._is_simple_node(d) or (self.reduce_parentheses and isinstance(d, c_ast.BinaryOp) and (self.precedence_map[d.op] > self.precedence_map[n.op]))))
    return '%s %s %s' % (lval_str, n.op, rval_str)
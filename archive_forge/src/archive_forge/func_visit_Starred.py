from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def visit_Starred(self, node):
    self._write('*')
    self.visit(node.value)
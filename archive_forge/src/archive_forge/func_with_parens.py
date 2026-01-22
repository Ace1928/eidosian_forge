from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def with_parens(f):

    def _f(self, node):
        self._write('(')
        f(self, node)
        self._write(')')
    return _f
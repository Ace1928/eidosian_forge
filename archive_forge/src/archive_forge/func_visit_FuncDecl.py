from . import c_ast
def visit_FuncDecl(self, n):
    return self._generate_type(n)
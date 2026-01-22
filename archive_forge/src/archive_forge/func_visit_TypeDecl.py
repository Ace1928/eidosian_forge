from . import c_ast
def visit_TypeDecl(self, n):
    return self._generate_type(n, emit_declname=False)
from . import c_ast
def visit_DeclList(self, n):
    s = self.visit(n.decls[0])
    if len(n.decls) > 1:
        s += ', ' + ', '.join((self.visit_Decl(decl, no_type=True) for decl in n.decls[1:]))
    return s
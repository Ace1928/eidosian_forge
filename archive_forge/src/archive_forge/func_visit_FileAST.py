from . import c_ast
def visit_FileAST(self, n):
    s = ''
    for ext in n.ext:
        if isinstance(ext, c_ast.FuncDef):
            s += self.visit(ext)
        elif isinstance(ext, c_ast.Pragma):
            s += self.visit(ext) + '\n'
        else:
            s += self.visit(ext) + ';\n'
    return s
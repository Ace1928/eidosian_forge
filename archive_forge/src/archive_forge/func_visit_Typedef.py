from . import c_ast
def visit_Typedef(self, n):
    s = ''
    if n.storage:
        s += ' '.join(n.storage) + ' '
    s += self._generate_type(n.type)
    return s
from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_initializer_list(self, p):
    """ initializer_list    : designation_opt initializer
                                | initializer_list COMMA designation_opt initializer
        """
    if len(p) == 3:
        init = p[2] if p[1] is None else c_ast.NamedInitializer(p[1], p[2])
        p[0] = c_ast.InitList([init], p[2].coord)
    else:
        init = p[4] if p[3] is None else c_ast.NamedInitializer(p[3], p[4])
        p[1].exprs.append(init)
        p[0] = p[1]
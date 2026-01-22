from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_constant_2(self, p):
    """ constant    : FLOAT_CONST
                        | HEX_FLOAT_CONST
        """
    if 'x' in p[1].lower():
        t = 'float'
    elif p[1][-1] in ('f', 'F'):
        t = 'float'
    elif p[1][-1] in ('l', 'L'):
        t = 'long double'
    else:
        t = 'double'
    p[0] = c_ast.Constant(t, p[1], self._token_coord(p, 1))
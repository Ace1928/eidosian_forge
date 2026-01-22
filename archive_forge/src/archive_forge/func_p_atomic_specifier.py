from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_atomic_specifier(self, p):
    """ atomic_specifier  : _ATOMIC LPAREN type_name RPAREN
        """
    typ = p[3]
    typ.quals.append('_Atomic')
    p[0] = typ
from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_type_qualifier(self, p):
    """ type_qualifier  : CONST
                            | RESTRICT
                            | VOLATILE
                            | _ATOMIC
        """
    p[0] = p[1]
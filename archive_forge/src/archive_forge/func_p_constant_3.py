from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_constant_3(self, p):
    """ constant    : CHAR_CONST
                        | WCHAR_CONST
                        | U8CHAR_CONST
                        | U16CHAR_CONST
                        | U32CHAR_CONST
        """
    p[0] = c_ast.Constant('char', p[1], self._token_coord(p, 1))
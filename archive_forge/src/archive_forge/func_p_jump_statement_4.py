from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_jump_statement_4(self, p):
    """ jump_statement  : RETURN expression SEMI
                            | RETURN SEMI
        """
    p[0] = c_ast.Return(p[2] if len(p) == 4 else None, self._token_coord(p, 1))
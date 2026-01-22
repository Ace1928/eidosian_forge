from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_unary_expression_3(self, p):
    """ unary_expression    : SIZEOF unary_expression
                                | SIZEOF LPAREN type_name RPAREN
                                | _ALIGNOF LPAREN type_name RPAREN
        """
    p[0] = c_ast.UnaryOp(p[1], p[2] if len(p) == 3 else p[3], self._token_coord(p, 1))
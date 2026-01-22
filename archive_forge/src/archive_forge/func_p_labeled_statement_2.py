from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_labeled_statement_2(self, p):
    """ labeled_statement : CASE constant_expression COLON pragmacomp_or_statement """
    p[0] = c_ast.Case(p[2], [p[4]], self._token_coord(p, 1))
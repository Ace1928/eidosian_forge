from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_conditional_expression(self, p):
    """ conditional_expression  : binary_expression
                                    | binary_expression CONDOP expression COLON conditional_expression
        """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = c_ast.TernaryOp(p[1], p[3], p[5], p[1].coord)
from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_assignment_expression(self, p):
    """ assignment_expression   : conditional_expression
                                    | unary_expression assignment_operator assignment_expression
        """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = c_ast.Assignment(p[2], p[1], p[3], p[1].coord)
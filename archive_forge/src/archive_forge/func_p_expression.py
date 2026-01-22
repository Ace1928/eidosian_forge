from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_expression(self, p):
    """ expression  : assignment_expression
                        | expression COMMA assignment_expression
        """
    if len(p) == 2:
        p[0] = p[1]
    else:
        if not isinstance(p[1], c_ast.ExprList):
            p[1] = c_ast.ExprList([p[1]], p[1].coord)
        p[1].exprs.append(p[3])
        p[0] = p[1]
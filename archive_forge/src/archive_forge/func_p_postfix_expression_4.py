from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_postfix_expression_4(self, p):
    """ postfix_expression  : postfix_expression PERIOD ID
                                | postfix_expression PERIOD TYPEID
                                | postfix_expression ARROW ID
                                | postfix_expression ARROW TYPEID
        """
    field = c_ast.ID(p[3], self._token_coord(p, 3))
    p[0] = c_ast.StructRef(p[1], p[2], field, p[1].coord)
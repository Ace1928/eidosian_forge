from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_primary_expression_5(self, p):
    """ primary_expression  : OFFSETOF LPAREN type_name COMMA offsetof_member_designator RPAREN
        """
    coord = self._token_coord(p, 1)
    p[0] = c_ast.FuncCall(c_ast.ID(p[1], coord), c_ast.ExprList([p[3], p[5]], coord), coord)
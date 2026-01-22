from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_enumerator(self, p):
    """ enumerator  : ID
                        | ID EQUALS constant_expression
        """
    if len(p) == 2:
        enumerator = c_ast.Enumerator(p[1], None, self._token_coord(p, 1))
    else:
        enumerator = c_ast.Enumerator(p[1], p[3], self._token_coord(p, 1))
    self._add_identifier(enumerator.name, enumerator.coord)
    p[0] = enumerator
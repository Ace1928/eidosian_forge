from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_pragmacomp_or_statement(self, p):
    """ pragmacomp_or_statement     : pppragma_directive statement
                                        | statement
        """
    if isinstance(p[1], c_ast.Pragma) and len(p) == 3:
        p[0] = c_ast.Compound(block_items=[p[1], p[2]], coord=self._token_coord(p, 1))
    else:
        p[0] = p[1]
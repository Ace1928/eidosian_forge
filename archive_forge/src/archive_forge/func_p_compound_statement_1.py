from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_compound_statement_1(self, p):
    """ compound_statement : brace_open block_item_list_opt brace_close """
    p[0] = c_ast.Compound(block_items=p[2], coord=self._token_coord(p, 1))
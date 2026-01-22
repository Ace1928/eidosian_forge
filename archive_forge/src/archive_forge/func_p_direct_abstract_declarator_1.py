from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_direct_abstract_declarator_1(self, p):
    """ direct_abstract_declarator  : LPAREN abstract_declarator RPAREN """
    p[0] = p[2]
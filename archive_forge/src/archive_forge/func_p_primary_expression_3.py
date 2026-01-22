from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_primary_expression_3(self, p):
    """ primary_expression  : unified_string_literal
                                | unified_wstring_literal
        """
    p[0] = p[1]
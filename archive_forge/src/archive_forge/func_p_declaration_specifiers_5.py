from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_declaration_specifiers_5(self, p):
    """ declaration_specifiers  : type_specifier
        """
    p[0] = self._add_declaration_specifier(None, p[1], 'type')
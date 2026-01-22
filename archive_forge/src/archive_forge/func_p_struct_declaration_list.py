from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_struct_declaration_list(self, p):
    """ struct_declaration_list     : struct_declaration
                                        | struct_declaration_list struct_declaration
        """
    if len(p) == 2:
        p[0] = p[1] or []
    else:
        p[0] = p[1] + (p[2] or [])
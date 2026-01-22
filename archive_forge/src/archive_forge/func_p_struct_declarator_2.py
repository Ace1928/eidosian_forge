from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_struct_declarator_2(self, p):
    """ struct_declarator   : declarator COLON constant_expression
                                | COLON constant_expression
        """
    if len(p) > 3:
        p[0] = {'decl': p[1], 'bitsize': p[3]}
    else:
        p[0] = {'decl': c_ast.TypeDecl(None, None, None, None), 'bitsize': p[2]}
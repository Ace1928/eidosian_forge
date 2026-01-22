from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_direct_abstract_declarator_7(self, p):
    """ direct_abstract_declarator  : LPAREN parameter_type_list_opt RPAREN
        """
    p[0] = c_ast.FuncDecl(args=p[2], type=c_ast.TypeDecl(None, None, None, None), coord=self._token_coord(p, 1))
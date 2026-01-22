from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_type_name(self, p):
    """ type_name   : specifier_qualifier_list abstract_declarator_opt
        """
    typename = c_ast.Typename(name='', quals=p[1]['qual'][:], align=None, type=p[2] or c_ast.TypeDecl(None, None, None, None), coord=self._token_coord(p, 2))
    p[0] = self._fix_decl_name_type(typename, p[1]['type'])
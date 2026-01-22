from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_direct_abstract_declarator_6(self, p):
    """ direct_abstract_declarator  : direct_abstract_declarator LPAREN parameter_type_list_opt RPAREN
        """
    func = c_ast.FuncDecl(args=p[3], type=None, coord=p[1].coord)
    p[0] = self._type_modify_decl(decl=p[1], modifier=func)
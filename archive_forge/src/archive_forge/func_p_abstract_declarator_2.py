from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_abstract_declarator_2(self, p):
    """ abstract_declarator     : pointer direct_abstract_declarator
        """
    p[0] = self._type_modify_decl(p[2], p[1])
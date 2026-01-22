from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_id_init_declarator(self, p):
    """ id_init_declarator : id_declarator
                               | id_declarator EQUALS initializer
        """
    p[0] = dict(decl=p[1], init=p[3] if len(p) > 2 else None)
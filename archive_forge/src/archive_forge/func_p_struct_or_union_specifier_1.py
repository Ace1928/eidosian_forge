from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_struct_or_union_specifier_1(self, p):
    """ struct_or_union_specifier   : struct_or_union ID
                                        | struct_or_union TYPEID
        """
    klass = self._select_struct_union_class(p[1])
    p[0] = klass(name=p[2], decls=None, coord=self._token_coord(p, 2))
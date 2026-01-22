from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_offsetof_member_designator(self, p):
    """ offsetof_member_designator : identifier
                                         | offsetof_member_designator PERIOD identifier
                                         | offsetof_member_designator LBRACKET expression RBRACKET
        """
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        p[0] = c_ast.StructRef(p[1], p[2], p[3], p[1].coord)
    elif len(p) == 5:
        p[0] = c_ast.ArrayRef(p[1], p[3], p[1].coord)
    else:
        raise NotImplementedError('Unexpected parsing state. len(p): %u' % len(p))
from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
@parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
def p_direct_xxx_declarator_4(self, p):
    """ direct_xxx_declarator   : direct_xxx_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET
                                    | direct_xxx_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET
        """
    listed_quals = [item if isinstance(item, list) else [item] for item in [p[3], p[4]]]
    dim_quals = [qual for sublist in listed_quals for qual in sublist if qual is not None]
    arr = c_ast.ArrayDecl(type=None, dim=p[5], dim_quals=dim_quals, coord=p[1].coord)
    p[0] = self._type_modify_decl(decl=p[1], modifier=arr)
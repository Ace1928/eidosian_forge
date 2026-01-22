from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
@parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
def p_direct_xxx_declarator_6(self, p):
    """ direct_xxx_declarator   : direct_xxx_declarator LPAREN parameter_type_list RPAREN
                                    | direct_xxx_declarator LPAREN identifier_list_opt RPAREN
        """
    func = c_ast.FuncDecl(args=p[3], type=None, coord=p[1].coord)
    if self._get_yacc_lookahead_token().type == 'LBRACE':
        if func.args is not None:
            for param in func.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    break
                self._add_identifier(param.name, param.coord)
    p[0] = self._type_modify_decl(decl=p[1], modifier=func)
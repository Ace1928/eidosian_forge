from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_parameter_declaration_2(self, p):
    """ parameter_declaration   : declaration_specifiers abstract_declarator_opt
        """
    spec = p[1]
    if not spec['type']:
        spec['type'] = [c_ast.IdentifierType(['int'], coord=self._token_coord(p, 1))]
    if len(spec['type']) > 1 and len(spec['type'][-1].names) == 1 and self._is_type_in_scope(spec['type'][-1].names[0]):
        decl = self._build_declarations(spec=spec, decls=[dict(decl=p[2], init=None)])[0]
    else:
        decl = c_ast.Typename(name='', quals=spec['qual'], align=None, type=p[2] or c_ast.TypeDecl(None, None, None, None), coord=self._token_coord(p, 2))
        typename = spec['type']
        decl = self._fix_decl_name_type(decl, typename)
    p[0] = decl
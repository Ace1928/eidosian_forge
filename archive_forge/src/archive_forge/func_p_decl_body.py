from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_decl_body(self, p):
    """ decl_body : declaration_specifiers init_declarator_list_opt
                      | declaration_specifiers_no_type id_init_declarator_list_opt
        """
    spec = p[1]
    if p[2] is None:
        ty = spec['type']
        s_u_or_e = (c_ast.Struct, c_ast.Union, c_ast.Enum)
        if len(ty) == 1 and isinstance(ty[0], s_u_or_e):
            decls = [c_ast.Decl(name=None, quals=spec['qual'], align=spec['alignment'], storage=spec['storage'], funcspec=spec['function'], type=ty[0], init=None, bitsize=None, coord=ty[0].coord)]
        else:
            decls = self._build_declarations(spec=spec, decls=[dict(decl=None, init=None)], typedef_namespace=True)
    else:
        decls = self._build_declarations(spec=spec, decls=p[2], typedef_namespace=True)
    p[0] = decls
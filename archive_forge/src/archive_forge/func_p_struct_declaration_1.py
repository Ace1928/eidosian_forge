from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_struct_declaration_1(self, p):
    """ struct_declaration : specifier_qualifier_list struct_declarator_list_opt SEMI
        """
    spec = p[1]
    assert 'typedef' not in spec['storage']
    if p[2] is not None:
        decls = self._build_declarations(spec=spec, decls=p[2])
    elif len(spec['type']) == 1:
        node = spec['type'][0]
        if isinstance(node, c_ast.Node):
            decl_type = node
        else:
            decl_type = c_ast.IdentifierType(node)
        decls = self._build_declarations(spec=spec, decls=[dict(decl=decl_type)])
    else:
        decls = self._build_declarations(spec=spec, decls=[dict(decl=None, init=None)])
    p[0] = decls
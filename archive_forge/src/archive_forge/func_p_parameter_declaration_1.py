from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def p_parameter_declaration_1(self, p):
    """ parameter_declaration   : declaration_specifiers id_declarator
                                    | declaration_specifiers typeid_noparen_declarator
        """
    spec = p[1]
    if not spec['type']:
        spec['type'] = [c_ast.IdentifierType(['int'], coord=self._token_coord(p, 1))]
    p[0] = self._build_declarations(spec=spec, decls=[dict(decl=p[2])])[0]
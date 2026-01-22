from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_definition(parser):
    if peek(parser, TokenKind.BRACE_L):
        return parse_operation_definition(parser)
    if peek(parser, TokenKind.NAME):
        name = parser.token.value
        if name in ('query', 'mutation', 'subscription'):
            return parse_operation_definition(parser)
        elif name == 'fragment':
            return parse_fragment_definition(parser)
        elif name in ('schema', 'scalar', 'type', 'interface', 'union', 'enum', 'input', 'extend', 'directive'):
            return parse_type_system_definition(parser)
    raise unexpected(parser)
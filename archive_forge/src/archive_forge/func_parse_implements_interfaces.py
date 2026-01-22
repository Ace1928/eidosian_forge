from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_implements_interfaces(parser):
    types = []
    if parser.token.value == 'implements':
        advance(parser)
        while True:
            types.append(parse_named_type(parser))
            if not peek(parser, TokenKind.NAME):
                break
    return types
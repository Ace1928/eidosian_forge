from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_operation_type(parser):
    operation_token = expect(parser, TokenKind.NAME)
    operation = operation_token.value
    if operation == 'query':
        return 'query'
    elif operation == 'mutation':
        return 'mutation'
    elif operation == 'subscription':
        return 'subscription'
    raise unexpected(parser, operation_token)
from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_variable_definitions(parser):
    if peek(parser, TokenKind.PAREN_L):
        return many(parser, TokenKind.PAREN_L, parse_variable_definition, TokenKind.PAREN_R)
    return []
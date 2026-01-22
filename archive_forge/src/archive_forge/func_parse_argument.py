from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_argument(parser):
    start = parser.token.start
    return ast.Argument(name=parse_name(parser), value=expect(parser, TokenKind.COLON) and parse_value_literal(parser, False), loc=loc(parser, start))
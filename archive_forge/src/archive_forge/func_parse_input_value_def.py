from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_input_value_def(parser):
    start = parser.token.start
    return ast.InputValueDefinition(name=parse_name(parser), type=expect(parser, TokenKind.COLON) and parse_type(parser), default_value=parse_const_value(parser) if skip(parser, TokenKind.EQUALS) else None, directives=parse_directives(parser), loc=loc(parser, start))
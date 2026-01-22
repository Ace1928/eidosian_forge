from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_directive_definition(parser):
    start = parser.token.start
    expect_keyword(parser, 'directive')
    expect(parser, TokenKind.AT)
    name = parse_name(parser)
    args = parse_argument_defs(parser)
    expect_keyword(parser, 'on')
    locations = parse_directive_locations(parser)
    return ast.DirectiveDefinition(name=name, locations=locations, arguments=args, loc=loc(parser, start))
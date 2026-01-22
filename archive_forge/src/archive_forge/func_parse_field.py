from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_field(parser):
    start = parser.token.start
    name_or_alias = parse_name(parser)
    if skip(parser, TokenKind.COLON):
        alias = name_or_alias
        name = parse_name(parser)
    else:
        alias = None
        name = name_or_alias
    return ast.Field(alias=alias, name=name, arguments=parse_arguments(parser), directives=parse_directives(parser), selection_set=parse_selection_set(parser) if peek(parser, TokenKind.BRACE_L) else None, loc=loc(parser, start))
from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_input_object_type_definition(parser):
    start = parser.token.start
    expect_keyword(parser, 'input')
    return ast.InputObjectTypeDefinition(name=parse_name(parser), directives=parse_directives(parser), fields=any(parser, TokenKind.BRACE_L, parse_input_value_def, TokenKind.BRACE_R), loc=loc(parser, start))
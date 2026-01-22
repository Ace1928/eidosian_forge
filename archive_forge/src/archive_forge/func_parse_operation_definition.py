from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_operation_definition(parser):
    start = parser.token.start
    if peek(parser, TokenKind.BRACE_L):
        return ast.OperationDefinition(operation='query', name=None, variable_definitions=None, directives=[], selection_set=parse_selection_set(parser), loc=loc(parser, start))
    operation = parse_operation_type(parser)
    name = None
    if peek(parser, TokenKind.NAME):
        name = parse_name(parser)
    return ast.OperationDefinition(operation=operation, name=name, variable_definitions=parse_variable_definitions(parser), directives=parse_directives(parser), selection_set=parse_selection_set(parser), loc=loc(parser, start))
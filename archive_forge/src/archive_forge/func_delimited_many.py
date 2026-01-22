from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def delimited_many(self, delimiter_kind: TokenKind, parse_fn: Callable[[], T]) -> List[T]:
    """Fetch many delimited nodes.

        Returns a non-empty list of parse nodes, determined by the ``parse_fn``. This
        list may begin with a lex token of ``delimiter_kind`` followed by items
        separated by lex tokens of ``delimiter_kind``. Advances the parser to the next
        lex token after the last item in the list.
        """
    expect_optional_token = partial(self.expect_optional_token, delimiter_kind)
    expect_optional_token()
    nodes: List[T] = []
    append = nodes.append
    while True:
        append(parse_fn())
        if not expect_optional_token():
            break
    return nodes
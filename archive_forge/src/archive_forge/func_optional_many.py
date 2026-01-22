from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def optional_many(self, open_kind: TokenKind, parse_fn: Callable[[], T], close_kind: TokenKind) -> List[T]:
    """Fetch matching nodes, maybe none.

        Returns a list of parse nodes, determined by the ``parse_fn``. It can be empty
        only if the open token is missing, otherwise it will always return a non-empty
        list that begins with a lex token of ``open_kind`` and ends with a lex token of
        ``close_kind``. Advances the parser to the next lex token after the closing
        token.
        """
    if self.expect_optional_token(open_kind):
        nodes = [parse_fn()]
        append = nodes.append
        expect_optional_token = partial(self.expect_optional_token, close_kind)
        while not expect_optional_token():
            append(parse_fn())
        return nodes
    return []
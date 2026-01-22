from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def expect_optional_token(self, kind: TokenKind) -> bool:
    """Expect the next token optionally to be of the given kind.

        If the next token is of the given kind, return True after advancing the lexer.
        Otherwise, do not change the parser state and return False.
        """
    token = self._lexer.token
    if token.kind == kind:
        self.advance_lexer()
        return True
    return False
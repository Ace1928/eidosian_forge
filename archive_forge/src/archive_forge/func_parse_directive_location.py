from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_directive_location(self) -> NameNode:
    """DirectiveLocation"""
    start = self._lexer.token
    name = self.parse_name()
    if name.value in DirectiveLocation.__members__:
        return name
    raise self.unexpected(start)
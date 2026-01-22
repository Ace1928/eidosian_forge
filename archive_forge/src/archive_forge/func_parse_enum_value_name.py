from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_enum_value_name(self) -> NameNode:
    """EnumValue: Name but not ``true``, ``false`` or ``null``"""
    if self._lexer.token.value in ('true', 'false', 'null'):
        raise GraphQLSyntaxError(self._lexer.source, self._lexer.token.start, f'{get_token_desc(self._lexer.token)} is reserved and cannot be used for an enum value.')
    return self.parse_name()
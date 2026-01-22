from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_float(self, _is_const: bool=False) -> FloatValueNode:
    token = self._lexer.token
    self.advance_lexer()
    return FloatValueNode(value=token.value, loc=self.loc(token))
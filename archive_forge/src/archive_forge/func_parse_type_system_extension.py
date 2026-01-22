from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_type_system_extension(self) -> TypeSystemExtensionNode:
    """TypeSystemExtension"""
    keyword_token = self._lexer.lookahead()
    if keyword_token.kind == TokenKind.NAME:
        method_name = self._parse_type_extension_method_names.get(cast(str, keyword_token.value))
        if method_name:
            return getattr(self, f'parse_{method_name}')()
    raise self.unexpected(keyword_token)
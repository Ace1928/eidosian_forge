from typing import List, NamedTuple, Optional
from ..error import GraphQLSyntaxError
from .ast import Token
from .block_string import dedent_block_string_lines
from .character_classes import is_digit, is_name_start, is_name_continue
from .source import Source
from .token_kind import TokenKind
def read_escaped_character(self, position: int) -> EscapeSequence:
    body = self.source.body
    value = _ESCAPED_CHARS.get(body[position + 1])
    if value:
        return EscapeSequence(value, 2)
    raise GraphQLSyntaxError(self.source, position, f"Invalid character escape sequence: '{body[position:position + 2]}'.")
from typing import List, NamedTuple, Optional
from ..error import GraphQLSyntaxError
from .ast import Token
from .block_string import dedent_block_string_lines
from .character_classes import is_digit, is_name_start, is_name_continue
from .source import Source
from .token_kind import TokenKind
def read_escaped_unicode_variable_width(self, position: int) -> EscapeSequence:
    body = self.source.body
    point = 0
    size = 3
    max_size = min(12, len(body) - position)
    while size < max_size:
        char = body[position + size]
        size += 1
        if char == '}':
            if size < 5 or not (0 <= point <= 55295 or 57344 <= point <= 1114111):
                break
            return EscapeSequence(chr(point), size)
        point = point << 4 | read_hex_digit(char)
        if point < 0:
            break
    raise GraphQLSyntaxError(self.source, position, f"Invalid Unicode escape sequence: '{body[position:position + size]}'.")
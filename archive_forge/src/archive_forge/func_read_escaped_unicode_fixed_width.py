from typing import List, NamedTuple, Optional
from ..error import GraphQLSyntaxError
from .ast import Token
from .block_string import dedent_block_string_lines
from .character_classes import is_digit, is_name_start, is_name_continue
from .source import Source
from .token_kind import TokenKind
def read_escaped_unicode_fixed_width(self, position: int) -> EscapeSequence:
    body = self.source.body
    code = read_16_bit_hex_code(body, position + 2)
    if 0 <= code <= 55295 or 57344 <= code <= 1114111:
        return EscapeSequence(chr(code), 6)
    if 55296 <= code <= 56319:
        if body[position + 6:position + 8] == '\\u':
            trailing_code = read_16_bit_hex_code(body, position + 8)
            if 56320 <= trailing_code <= 57343:
                return EscapeSequence((chr(code) + chr(trailing_code)).encode('utf-16', 'surrogatepass').decode('utf-16'), 12)
    raise GraphQLSyntaxError(self.source, position, f"Invalid Unicode escape sequence: '{body[position:position + 6]}'.")
from typing import List, NamedTuple, Optional
from ..error import GraphQLSyntaxError
from .ast import Token
from .block_string import dedent_block_string_lines
from .character_classes import is_digit, is_name_start, is_name_continue
from .source import Source
from .token_kind import TokenKind
def print_code_point_at(self, location: int) -> str:
    """Print the code point at the given location.

        Prints the code point (or end of file reference) at a given location in a
        source for use in error messages.

        Printable ASCII is printed quoted, while other points are printed in Unicode
        code point form (ie. U+1234).
        """
    body = self.source.body
    if location >= len(body):
        return TokenKind.EOF.value
    char = body[location]
    if ' ' <= char <= '~':
        return '\'"\'' if char == '"' else f"'{char}'"
    point = ord(body[location:location + 2].encode('utf-16', 'surrogatepass').decode('utf-16') if is_supplementary_code_point(body, location) else char)
    return f'U+{point:04X}'
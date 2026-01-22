from typing import List, NamedTuple, Optional
from ..error import GraphQLSyntaxError
from .ast import Token
from .block_string import dedent_block_string_lines
from .character_classes import is_digit, is_name_start, is_name_continue
from .source import Source
from .token_kind import TokenKind
def read_block_string(self, start: int) -> Token:
    """Read a block string token from the source file."""
    body = self.source.body
    body_length = len(body)
    line_start = self.line_start
    position = start + 3
    chunk_start = position
    current_line = ''
    block_lines = []
    while position < body_length:
        char = body[position]
        if char == '"' and body[position + 1:position + 3] == '""':
            current_line += body[chunk_start:position]
            block_lines.append(current_line)
            token = self.create_token(TokenKind.BLOCK_STRING, start, position + 3, '\n'.join(dedent_block_string_lines(block_lines)))
            self.line += len(block_lines) - 1
            self.line_start = line_start
            return token
        if char == '\\' and body[position + 1:position + 4] == '"""':
            current_line += body[chunk_start:position]
            chunk_start = position + 1
            position += 4
            continue
        if char in '\r\n':
            current_line += body[chunk_start:position]
            block_lines.append(current_line)
            if char == '\r' and body[position + 1:position + 2] == '\n':
                position += 2
            else:
                position += 1
            current_line = ''
            chunk_start = line_start = position
            continue
        if is_unicode_scalar_value(char):
            position += 1
        elif is_supplementary_code_point(body, position):
            position += 2
        else:
            raise GraphQLSyntaxError(self.source, position, f'Invalid character within String: {self.print_code_point_at(position)}.')
    raise GraphQLSyntaxError(self.source, position, 'Unterminated string.')
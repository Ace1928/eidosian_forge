from typing import List, NamedTuple, Optional
from ..error import GraphQLSyntaxError
from .ast import Token
from .block_string import dedent_block_string_lines
from .character_classes import is_digit, is_name_start, is_name_continue
from .source import Source
from .token_kind import TokenKind
def read_next_token(self, start: int) -> Token:
    """Get the next token from the source starting at the given position.

        This skips over whitespace until it finds the next lexable token, then lexes
        punctuators immediately or calls the appropriate helper function for more
        complicated tokens.
        """
    body = self.source.body
    body_length = len(body)
    position = start
    while position < body_length:
        char = body[position]
        if char in ' \t,\ufeff':
            position += 1
            continue
        elif char == '\n':
            position += 1
            self.line += 1
            self.line_start = position
            continue
        elif char == '\r':
            if body[position + 1:position + 2] == '\n':
                position += 2
            else:
                position += 1
            self.line += 1
            self.line_start = position
            continue
        if char == '#':
            return self.read_comment(position)
        if char == '"':
            if body[position + 1:position + 3] == '""':
                return self.read_block_string(position)
            return self.read_string(position)
        kind = _KIND_FOR_PUNCT.get(char)
        if kind:
            return self.create_token(kind, position, position + 1)
        if is_digit(char) or char == '-':
            return self.read_number(position, char)
        if is_name_start(char):
            return self.read_name(position)
        if char == '.':
            if body[position + 1:position + 3] == '..':
                return self.create_token(TokenKind.SPREAD, position, position + 3)
        message = 'Unexpected single quote character (\'), did you mean to use a double quote (")?' if char == "'" else f'Unexpected character: {self.print_code_point_at(position)}.' if is_unicode_scalar_value(char) or is_supplementary_code_point(body, position) else f'Invalid character: {self.print_code_point_at(position)}.'
        raise GraphQLSyntaxError(self.source, position, message)
    return self.create_token(TokenKind.EOF, body_length, body_length)
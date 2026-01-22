import json
from ..error import GraphQLSyntaxError
def read_name(source, position):
    """Reads an alphanumeric + underscore name from the source.

    [_A-Za-z][_0-9A-Za-z]*"""
    body = source.body
    body_length = len(body)
    end = position + 1
    while end != body_length:
        code = char_code_at(body, end)
        if not (code is not None and (code == 95 or 48 <= code <= 57 or 65 <= code <= 90 or (97 <= code <= 122))):
            break
        end += 1
    return Token(TokenKind.NAME, position, end, body[position:end])
import json
from ..error import GraphQLSyntaxError
def read_number(source, start, first_code):
    """Reads a number token from the source file, either a float
    or an int depending on whether a decimal point appears.
    """
    code = first_code
    body = source.body
    position = start
    is_float = False
    if code == 45:
        position += 1
        code = char_code_at(body, position)
    if code == 48:
        position += 1
        code = char_code_at(body, position)
        if code is not None and 48 <= code <= 57:
            raise GraphQLSyntaxError(source, position, u'Invalid number, unexpected digit after 0: {}.'.format(print_char_code(code)))
    else:
        position = read_digits(source, position, code)
        code = char_code_at(body, position)
    if code == 46:
        is_float = True
        position += 1
        code = char_code_at(body, position)
        position = read_digits(source, position, code)
        code = char_code_at(body, position)
    if code in (69, 101):
        is_float = True
        position += 1
        code = char_code_at(body, position)
        if code in (43, 45):
            position += 1
            code = char_code_at(body, position)
        position = read_digits(source, position, code)
    return Token(TokenKind.FLOAT if is_float else TokenKind.INT, start, position, body[start:position])
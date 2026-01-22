import json
from ..error import GraphQLSyntaxError
def read_string(source, start):
    """Reads a string token from the source file.

    "([^"\\
\r\u2028\u2029]|(\\(u[0-9a-fA-F]{4}|["\\/bfnrt])))*"
    """
    body = source.body
    body_length = len(body)
    position = start + 1
    chunk_start = position
    code = 0
    value = []
    append = value.append
    while position < body_length:
        code = char_code_at(body, position)
        if not (code is not None and code not in (10, 13, 34)):
            break
        if code < 32 and code != 9:
            raise GraphQLSyntaxError(source, position, u'Invalid character within String: {}.'.format(print_char_code(code)))
        position += 1
        if code == 92:
            append(body[chunk_start:position - 1])
            code = char_code_at(body, position)
            escaped = ESCAPED_CHAR_CODES.get(code)
            if escaped is not None:
                append(escaped)
            elif code == 117:
                char_code = uni_char_code(char_code_at(body, position + 1) or 0, char_code_at(body, position + 2) or 0, char_code_at(body, position + 3) or 0, char_code_at(body, position + 4) or 0)
                if char_code < 0:
                    raise GraphQLSyntaxError(source, position, u'Invalid character escape sequence: \\u{}.'.format(body[position + 1:position + 5]))
                append(chr(char_code))
                position += 4
            else:
                raise GraphQLSyntaxError(source, position, u'Invalid character escape sequence: \\{}.'.format(chr(code)))
            position += 1
            chunk_start = position
    if code != 34:
        raise GraphQLSyntaxError(source, position, 'Unterminated string')
    append(body[chunk_start:position])
    return Token(TokenKind.STRING, start, position + 1, u''.join(value))
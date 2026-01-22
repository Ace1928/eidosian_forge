import struct
def raiseBadHeader(field, expected):
    actual = int(headersMap[field])
    message = f'The header {field} is expected to be 0, not {actual}'
    if actual != expected:
        raise ParserError(message)
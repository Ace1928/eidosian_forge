import json
from ..error import GraphQLSyntaxError
def print_char_code(code):
    if code is None:
        return '<EOF>'
    if code < 127:
        return json.dumps(chr(code))
    return '"\\u%04X"' % code
from ..language.location import get_location
from .base import GraphQLError
def highlight_source_at_location(source, location):
    line = location.line
    lines = source.body.splitlines()
    pad_len = len(str(line + 1))
    result = u''
    format = (u'{:>' + str(pad_len) + '}: {}\n').format
    if line >= 2:
        result += format(line - 1, lines[line - 2])
    result += format(line, lines[line - 1])
    result += ' ' * (1 + pad_len + location.column) + '^\n'
    if line < len(lines):
        result += format(line + 1, lines[line])
    return result
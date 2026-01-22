from __future__ import (absolute_import, division, print_function)
def parse_quoted_value(cur):
    if cur == '\\':
        parser.next()
        if parser.done():
            raise InvalidLogFmt('Unterminated escape sequence in quoted string')
        cur = parser.cur()
        if cur in _ESCAPE_DICT:
            value.append(_ESCAPE_DICT[cur])
        elif cur != 'u':
            raise InvalidLogFmt('Unknown escape sequence {seq!r}'.format(seq='\\' + cur))
        else:
            parser.prev()
            value.append(parser.parse_unicode_sequence())
        parser.next()
        return _Mode.QUOTED_VALUE
    elif cur == '"':
        handle_kv()
        parser.next()
        return _Mode.GARBAGE
    elif cur < ' ':
        raise InvalidLogFmt('Control characters in quoted string are not allowed')
    else:
        value.append(cur)
        parser.next()
        return _Mode.QUOTED_VALUE
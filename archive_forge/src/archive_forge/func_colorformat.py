from pygments.token import Token, STANDARD_TYPES
from pygments.util import add_metaclass
def colorformat(text):
    if text in ansicolors:
        return text
    if text[0:1] == '#':
        col = text[1:]
        if len(col) == 6:
            return col
        elif len(col) == 3:
            return col[0] * 2 + col[1] * 2 + col[2] * 2
    elif text == '':
        return ''
    assert False, 'wrong color format %r' % text
from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on('<Esc+t>')
def transpose_word_before_cursor(cursor_offset, line):
    return (cursor_offset, line)
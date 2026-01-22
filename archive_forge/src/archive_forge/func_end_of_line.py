from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on(config='end_of_line_key')
@edit_keys.on('<END>')
def end_of_line(cursor_offset, line):
    return (len(line), line)
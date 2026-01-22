from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on('<Esc+y>')
def yank_prev_prev_killed_text(cursor_offset, line, cut_buffer):
    return (cursor_offset + len(cut_buffer), line[:cursor_offset] + cut_buffer + line[cursor_offset:])
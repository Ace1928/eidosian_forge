from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
@edit_keys.on('<BACKSPACE>')
@edit_keys.on(config='backspace_key')
def backspace(cursor_offset, line):
    if cursor_offset == 0:
        return (cursor_offset, line)
    if not line[:cursor_offset].strip():
        to_delete = (cursor_offset - 1) % INDENT + 1
        return (cursor_offset - to_delete, line[:cursor_offset - to_delete] + line[cursor_offset:])
    on_closing_char, pair_close = cursor_on_closing_char_pair(cursor_offset, line)
    if on_closing_char and pair_close:
        return (cursor_offset - 1, line[:cursor_offset - 1] + line[cursor_offset + 1:])
    return (cursor_offset - 1, line[:cursor_offset - 1] + line[cursor_offset:])
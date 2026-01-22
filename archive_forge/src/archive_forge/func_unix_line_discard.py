from __future__ import unicode_literals
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER
from prompt_toolkit.selection import PasteMode
from six.moves import range
import six
from .completion import generate_completions, display_completions_like_readline
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.input_processor import KeyPress
from prompt_toolkit.keys import Keys
@register('unix-line-discard')
def unix_line_discard(event):
    """
    Kill backward from the cursor to the beginning of the current line.
    """
    buff = event.current_buffer
    if buff.document.cursor_position_col == 0 and buff.document.cursor_position > 0:
        buff.delete_before_cursor(count=1)
    else:
        deleted = buff.delete_before_cursor(count=-buff.document.get_start_of_line_position())
        event.cli.clipboard.set_text(deleted)
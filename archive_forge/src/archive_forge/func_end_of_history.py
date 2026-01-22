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
@register('end-of-history')
def end_of_history(event):
    """
    Move to the end of the input history, i.e., the line currently being entered.
    """
    event.current_buffer.history_forward(count=10 ** 100)
    buff = event.current_buffer
    buff.go_to_history(len(buff._working_lines) - 1)
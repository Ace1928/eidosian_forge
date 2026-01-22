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
@register('reverse-search-history')
def reverse_search_history(event):
    """
    Search backward starting at the current line and moving `up' through
    the history as necessary. This is an incremental search.
    """
    event.cli.current_search_state.direction = IncrementalSearchDirection.BACKWARD
    event.cli.push_focus(SEARCH_BUFFER)
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
@register('edit-and-execute-command')
def edit_and_execute(event):
    """
    Invoke an editor on the current command line, and accept the result.
    """
    buff = event.current_buffer
    buff.open_in_editor(event.cli)
    buff.accept_action.validate_and_handle(event.cli, buff)
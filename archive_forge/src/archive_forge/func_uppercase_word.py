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
@register('uppercase-word')
def uppercase_word(event):
    """
    Uppercase the current (or following) word.
    """
    buff = event.current_buffer
    for i in range(event.arg):
        pos = buff.document.find_next_word_ending()
        words = buff.document.text_after_cursor[:pos]
        buff.insert_text(words.upper(), overwrite=True)
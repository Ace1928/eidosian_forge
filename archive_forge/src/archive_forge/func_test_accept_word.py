import pytest
from IPython.terminal.shortcuts.auto_suggest import (
from IPython.terminal.shortcuts.auto_match import skip_over
from IPython.terminal.shortcuts import create_ipython_shortcuts, reset_search_buffer
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.enums import DEFAULT_BUFFER
from unittest.mock import patch, Mock
@pytest.mark.parametrize('text, suggestion, expected', [('', 'def out(tag: str, n=50):', 'def '), ('d', 'ef out(tag: str, n=50):', 'ef '), ('de', 'f out(tag: str, n=50):', 'f '), ('def', ' out(tag: str, n=50):', ' '), ('def ', 'out(tag: str, n=50):', 'out(tag: ')])
def test_accept_word(text, suggestion, expected):
    event = make_event(text, len(text), suggestion)
    event.current_buffer.insert_text = Mock()
    accept_word(event)
    assert event.current_buffer.insert_text.called
    assert event.current_buffer.insert_text.call_args[0] == (expected,)
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
@pytest.mark.parametrize('text, suggestion, expected, cursor', [('', 'def out(tag: str, n=50):', 'def out(tag: str, n=50):', 0), ('def ', 'out(tag: str, n=50):', 'out(tag: str, n=50):', 4)])
def test_accept_and_keep_cursor(text, suggestion, expected, cursor):
    event = make_event(text, cursor, suggestion)
    buffer = event.current_buffer
    buffer.insert_text = Mock()
    accept_and_keep_cursor(event)
    assert buffer.insert_text.called
    assert buffer.insert_text.call_args[0] == (expected,)
    assert buffer.cursor_position == cursor
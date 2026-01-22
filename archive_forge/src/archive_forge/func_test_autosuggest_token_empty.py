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
def test_autosuggest_token_empty():
    full = 'def out(tag: str, n=50):'
    event = make_event(full, len(full), '')
    event.current_buffer.insert_text = Mock()
    with patch('prompt_toolkit.key_binding.bindings.named_commands.forward_word') as forward_word:
        accept_token(event)
        assert not event.current_buffer.insert_text.called
        assert forward_word.called
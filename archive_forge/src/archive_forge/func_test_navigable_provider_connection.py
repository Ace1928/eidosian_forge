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
def test_navigable_provider_connection():
    provider = NavigableAutoSuggestFromHistory()
    provider.skip_lines = 1
    session_1 = create_session_mock()
    provider.connect(session_1)
    assert provider.skip_lines == 1
    session_1.default_buffer.on_text_insert.fire()
    assert provider.skip_lines == 0
    session_2 = create_session_mock()
    provider.connect(session_2)
    provider.skip_lines = 2
    assert provider.skip_lines == 2
    session_2.default_buffer.on_text_insert.fire()
    assert provider.skip_lines == 0
    provider.skip_lines = 3
    provider.disconnect()
    session_1.default_buffer.on_text_insert.fire()
    session_2.default_buffer.on_text_insert.fire()
    assert provider.skip_lines == 3
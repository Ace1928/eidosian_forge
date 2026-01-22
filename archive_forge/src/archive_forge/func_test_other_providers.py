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
def test_other_providers():
    """Ensure that swapping autosuggestions does not break with other providers"""
    provider = AutoSuggestFromHistory()
    ip = get_ipython()
    ip.auto_suggest = provider
    event = Mock()
    event.current_buffer = Buffer()
    assert swap_autosuggestion_up(event) is None
    assert swap_autosuggestion_down(event) is None
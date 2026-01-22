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
def test_disable_shortcut(ipython_with_prompt):
    matched = find_bindings_by_command(accept_token)
    assert len(matched) == 1
    ipython_with_prompt.shortcuts = [{'command': 'IPython:auto_suggest.accept_token', 'new_keys': []}]
    matched = find_bindings_by_command(accept_token)
    assert len(matched) == 0
    ipython_with_prompt.shortcuts = []
    matched = find_bindings_by_command(accept_token)
    assert len(matched) == 1
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
def test_modify_shortcut_with_filters(ipython_with_prompt):
    matched = find_bindings_by_command(skip_over)
    matched_keys = {m.keys[0] for m in matched}
    assert matched_keys == {')', ']', '}', "'", '"'}
    with pytest.raises(ValueError, match='Multiple shortcuts matching'):
        ipython_with_prompt.shortcuts = [{'command': 'IPython:auto_match.skip_over', 'new_keys': ['x']}]
    ipython_with_prompt.shortcuts = [{'command': 'IPython:auto_match.skip_over', 'new_keys': ['x'], 'match_filter': 'focused_insert & auto_match & followed_by_single_quote'}]
    matched = find_bindings_by_command(skip_over)
    matched_keys = {m.keys[0] for m in matched}
    assert matched_keys == {')', ']', '}', 'x', '"'}
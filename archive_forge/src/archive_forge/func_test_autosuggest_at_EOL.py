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
@pytest.mark.parametrize('text, cursor, suggestion, called', [('123456', 6, '123456789', True), ('123456', 3, '123456789', False), ('123456   \n789', 6, '123456789', True)])
def test_autosuggest_at_EOL(text, cursor, suggestion, called):
    """
    test that autosuggest is only applied at end of line.
    """
    event = make_event(text, cursor, suggestion)
    event.current_buffer.insert_text = Mock()
    accept_or_jump_to_end(event)
    if called:
        event.current_buffer.insert_text.assert_called()
    else:
        event.current_buffer.insert_text.assert_not_called()
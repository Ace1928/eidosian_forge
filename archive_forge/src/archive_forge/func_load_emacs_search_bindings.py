from __future__ import unicode_literals
from prompt_toolkit.buffer import SelectionType, indent, unindent
from prompt_toolkit.keys import Keys
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import Condition, EmacsMode, HasSelection, EmacsInsertMode, HasFocus, HasArg
from prompt_toolkit.completion import CompleteEvent
from .scroll import scroll_page_up, scroll_page_down
from .named_commands import get_by_name
from ..registry import Registry, ConditionalRegistry
def load_emacs_search_bindings(get_search_state=None):
    registry = ConditionalRegistry(Registry(), EmacsMode())
    handle = registry.add_binding
    has_focus = HasFocus(SEARCH_BUFFER)
    assert get_search_state is None or callable(get_search_state)
    if not get_search_state:

        def get_search_state(cli):
            return cli.search_state

    @handle(Keys.ControlG, filter=has_focus)
    @handle(Keys.ControlC, filter=has_focus)
    def _(event):
        """
        Abort an incremental search and restore the original line.
        """
        search_buffer = event.cli.buffers[SEARCH_BUFFER]
        search_buffer.reset()
        event.cli.pop_focus()

    @handle(Keys.ControlJ, filter=has_focus)
    def _(event):
        """
        When enter pressed in isearch, quit isearch mode. (Multiline
        isearch would be too complicated.)
        """
        input_buffer = event.cli.buffers.previous(event.cli)
        search_buffer = event.cli.buffers[SEARCH_BUFFER]
        if search_buffer.text:
            get_search_state(event.cli).text = search_buffer.text
        input_buffer.apply_search(get_search_state(event.cli), include_current_position=True)
        search_buffer.append_to_history()
        search_buffer.reset()
        event.cli.pop_focus()

    @handle(Keys.ControlR, filter=~has_focus)
    def _(event):
        get_search_state(event.cli).direction = IncrementalSearchDirection.BACKWARD
        event.cli.push_focus(SEARCH_BUFFER)

    @handle(Keys.ControlS, filter=~has_focus)
    def _(event):
        get_search_state(event.cli).direction = IncrementalSearchDirection.FORWARD
        event.cli.push_focus(SEARCH_BUFFER)

    def incremental_search(cli, direction, count=1):
        """ Apply search, but keep search buffer focussed. """
        search_state = get_search_state(cli)
        direction_changed = search_state.direction != direction
        search_state.text = cli.buffers[SEARCH_BUFFER].text
        search_state.direction = direction
        if not direction_changed:
            input_buffer = cli.buffers.previous(cli)
            input_buffer.apply_search(search_state, include_current_position=False, count=count)

    @handle(Keys.ControlR, filter=has_focus)
    @handle(Keys.Up, filter=has_focus)
    def _(event):
        incremental_search(event.cli, IncrementalSearchDirection.BACKWARD, count=event.arg)

    @handle(Keys.ControlS, filter=has_focus)
    @handle(Keys.Down, filter=has_focus)
    def _(event):
        incremental_search(event.cli, IncrementalSearchDirection.FORWARD, count=event.arg)
    return registry
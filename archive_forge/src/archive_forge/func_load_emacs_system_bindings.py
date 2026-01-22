from __future__ import unicode_literals
from prompt_toolkit.buffer import SelectionType, indent, unindent
from prompt_toolkit.keys import Keys
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import Condition, EmacsMode, HasSelection, EmacsInsertMode, HasFocus, HasArg
from prompt_toolkit.completion import CompleteEvent
from .scroll import scroll_page_up, scroll_page_down
from .named_commands import get_by_name
from ..registry import Registry, ConditionalRegistry
def load_emacs_system_bindings():
    registry = ConditionalRegistry(Registry(), EmacsMode())
    handle = registry.add_binding
    has_focus = HasFocus(SYSTEM_BUFFER)

    @handle(Keys.Escape, '!', filter=~has_focus)
    def _(event):
        """
        M-'!' opens the system prompt.
        """
        event.cli.push_focus(SYSTEM_BUFFER)

    @handle(Keys.Escape, filter=has_focus)
    @handle(Keys.ControlG, filter=has_focus)
    @handle(Keys.ControlC, filter=has_focus)
    def _(event):
        """
        Cancel system prompt.
        """
        event.cli.buffers[SYSTEM_BUFFER].reset()
        event.cli.pop_focus()

    @handle(Keys.ControlJ, filter=has_focus)
    def _(event):
        """
        Run system command.
        """
        system_line = event.cli.buffers[SYSTEM_BUFFER]
        event.cli.run_system_command(system_line.text)
        system_line.reset(append_to_history=True)
        event.cli.pop_focus()
    return registry
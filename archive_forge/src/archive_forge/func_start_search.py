from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING
from .application.current import get_app
from .filters import FilterOrBool, is_searching, to_filter
from .key_binding.vi_state import InputMode
def start_search(buffer_control: BufferControl | None=None, direction: SearchDirection=SearchDirection.FORWARD) -> None:
    """
    Start search through the given `buffer_control` using the
    `search_buffer_control`.

    :param buffer_control: Start search for this `BufferControl`. If not given,
        search through the current control.
    """
    from prompt_toolkit.layout.controls import BufferControl
    assert buffer_control is None or isinstance(buffer_control, BufferControl)
    layout = get_app().layout
    if buffer_control is None:
        if not isinstance(layout.current_control, BufferControl):
            return
        buffer_control = layout.current_control
    search_buffer_control = buffer_control.search_buffer_control
    if search_buffer_control:
        buffer_control.search_state.direction = direction
        layout.focus(search_buffer_control)
        layout.search_links[search_buffer_control] = buffer_control
        get_app().vi_state.input_mode = InputMode.INSERT
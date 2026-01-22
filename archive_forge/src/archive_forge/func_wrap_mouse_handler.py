from __future__ import annotations
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent
from .containers import Container, ScrollOffsets
from .dimension import AnyDimension, Dimension, sum_layout_dimensions, to_dimension
from .mouse_handlers import MouseHandler, MouseHandlers
from .screen import Char, Screen, WritePosition
def wrap_mouse_handler(handler: MouseHandler) -> MouseHandler:
    """Wrap mouse handler. Translate coordinates in `MouseEvent`."""
    if handler not in mouse_handler_wrappers:

        def new_handler(event: MouseEvent) -> None:
            new_event = MouseEvent(position=Point(x=event.position.x - xpos, y=event.position.y + self.vertical_scroll - ypos), event_type=event.event_type, button=event.button, modifiers=event.modifiers)
            handler(new_event)
        mouse_handler_wrappers[handler] = new_handler
    return mouse_handler_wrappers[handler]
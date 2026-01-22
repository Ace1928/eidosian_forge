from __future__ import annotations
from prompt_toolkit.filters import buffer_has_focus, emacs_mode, vi_mode
from prompt_toolkit.key_binding.key_bindings import (
from .scroll import (
def load_vi_page_navigation_bindings() -> KeyBindingsBase:
    """
    Key bindings, for scrolling up and down through pages.
    This are separate bindings, because GNU readline doesn't have them.
    """
    key_bindings = KeyBindings()
    handle = key_bindings.add
    handle('c-f')(scroll_forward)
    handle('c-b')(scroll_backward)
    handle('c-d')(scroll_half_page_down)
    handle('c-u')(scroll_half_page_up)
    handle('c-e')(scroll_one_line_down)
    handle('c-y')(scroll_one_line_up)
    handle('pagedown')(scroll_page_down)
    handle('pageup')(scroll_page_up)
    return ConditionalKeyBindings(key_bindings, vi_mode)
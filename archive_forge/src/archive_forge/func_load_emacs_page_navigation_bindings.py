from __future__ import annotations
from prompt_toolkit.filters import buffer_has_focus, emacs_mode, vi_mode
from prompt_toolkit.key_binding.key_bindings import (
from .scroll import (
def load_emacs_page_navigation_bindings() -> KeyBindingsBase:
    """
    Key bindings, for scrolling up and down through pages.
    This are separate bindings, because GNU readline doesn't have them.
    """
    key_bindings = KeyBindings()
    handle = key_bindings.add
    handle('c-v')(scroll_page_down)
    handle('pagedown')(scroll_page_down)
    handle('escape', 'v')(scroll_page_up)
    handle('pageup')(scroll_page_up)
    return ConditionalKeyBindings(key_bindings, emacs_mode)
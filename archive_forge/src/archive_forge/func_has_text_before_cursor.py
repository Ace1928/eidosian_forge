from __future__ import annotations
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import (
from prompt_toolkit.key_binding.key_processor import KeyPress, KeyPressEvent
from prompt_toolkit.keys import Keys
from ..key_bindings import KeyBindings
from .named_commands import get_by_name
@Condition
def has_text_before_cursor() -> bool:
    return bool(get_app().current_buffer.text)
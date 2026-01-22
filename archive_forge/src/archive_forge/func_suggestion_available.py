from __future__ import annotations
import re
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition, emacs_mode
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
@Condition
def suggestion_available() -> bool:
    app = get_app()
    return app.current_buffer.suggestion is not None and len(app.current_buffer.suggestion.text) > 0 and app.current_buffer.document.is_cursor_at_the_end
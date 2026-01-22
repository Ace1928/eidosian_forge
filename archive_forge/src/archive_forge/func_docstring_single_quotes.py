import re
from prompt_toolkit.key_binding import KeyPressEvent
def docstring_single_quotes(event: KeyPressEvent):
    """Auto-close docstring (single quotes)"""
    event.current_buffer.insert_text("''''")
    event.current_buffer.cursor_left(3)
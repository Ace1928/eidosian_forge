import re
from prompt_toolkit.key_binding import KeyPressEvent
def double_quote(event: KeyPressEvent):
    """Auto-close double quotes"""
    event.current_buffer.insert_text('""')
    event.current_buffer.cursor_left()
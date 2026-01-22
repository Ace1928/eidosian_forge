import re
from prompt_toolkit.key_binding import KeyPressEvent
def parenthesis(event: KeyPressEvent):
    """Auto-close parenthesis"""
    event.current_buffer.insert_text('()')
    event.current_buffer.cursor_left()
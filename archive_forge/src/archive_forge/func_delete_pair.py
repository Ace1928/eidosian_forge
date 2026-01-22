import re
from prompt_toolkit.key_binding import KeyPressEvent
def delete_pair(event: KeyPressEvent):
    """Delete auto-closed parenthesis"""
    event.current_buffer.delete()
    event.current_buffer.delete_before_cursor()
from __future__ import annotations
from collections import defaultdict
from ..key_binding import KeyPress
from .base import Input
def get_typeahead(input_obj: Input) -> list[KeyPress]:
    """
    Retrieve typeahead and reset the buffer for this input.
    """
    global _buffer
    key = input_obj.typeahead_hash()
    result = _buffer[key]
    _buffer[key] = []
    return result
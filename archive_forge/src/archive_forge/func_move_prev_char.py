from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def move_prev_char(text: str | bytes, start_offs: int, end_offs: int) -> int:
    """
    Return the position of the character before end_offs.
    """
    if start_offs >= end_offs:
        raise ValueError((start_offs, end_offs))
    if isinstance(text, str):
        return end_offs - 1
    if not isinstance(text, bytes):
        raise TypeError(text)
    if _byte_encoding == 'utf8':
        o = end_offs - 1
        while text[o] & 192 == 128:
            o -= 1
        return o
    if _byte_encoding == 'wide' and within_double_byte(text, start_offs, end_offs - 1) == 2:
        return end_offs - 2
    return end_offs - 1
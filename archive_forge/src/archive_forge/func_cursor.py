import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def cursor(s):
    """'ab|c' -> (2, 'abc')"""
    cursor_offset = s.index('|')
    line = s[:cursor_offset] + s[cursor_offset + 1:]
    return (cursor_offset, line)
import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def term_len(x: str) -> int:
    return len(strip_ansi(x))
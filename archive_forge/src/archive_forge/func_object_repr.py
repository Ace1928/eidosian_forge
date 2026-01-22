from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
def object_repr(self, obj: type[dict] | t.Callable | type[list] | None) -> str:
    r = repr(obj)
    return f'<span class="object">{escape(r)}</span>'
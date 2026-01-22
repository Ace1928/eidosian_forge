from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
def repr(self, obj: object) -> str:
    recursive = False
    for item in self._stack:
        if item is obj:
            recursive = True
            break
    self._stack.append(obj)
    try:
        try:
            return self.dispatch_repr(obj, recursive)
        except Exception:
            return self.fallback_repr()
    finally:
        self._stack.pop()
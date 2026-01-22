import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def get_type_comment(self, node):
    comment = self._type_ignores.get(node.lineno) or node.type_comment
    if comment is not None:
        return f' # type: {comment}'
from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
def line_for_node(self, node: ast.AST) -> TLineNo:
    """What is the right line number to use for this node?

        This dispatches to _line__Node functions where needed.

        """
    node_name = node.__class__.__name__
    handler = cast(Optional[Callable[[ast.AST], TLineNo]], getattr(self, '_line__' + node_name, None))
    if handler is not None:
        return handler(node)
    else:
        return node.lineno
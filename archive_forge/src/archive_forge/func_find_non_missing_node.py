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
def find_non_missing_node(self, node: ast.AST) -> ast.AST | None:
    """Search `node` looking for a child that has not been optimized away.

        This might return the node you started with, or it will work recursively
        to find a child node in self.statements.

        Returns a node, or None if none of the node remains.

        """
    lineno = self.line_for_node(node)
    first_line = self.multiline.get(lineno, lineno)
    if first_line in self.statements:
        return node
    missing_fn = cast(Optional[Callable[[ast.AST], Optional[ast.AST]]], getattr(self, '_missing__' + node.__class__.__name__, None))
    if missing_fn is not None:
        ret_node = missing_fn(node)
    else:
        ret_node = None
    return ret_node
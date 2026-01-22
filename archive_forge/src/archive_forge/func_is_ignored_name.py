from __future__ import annotations
import ast
import builtins
import sys
import typing
from ast import (
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload
def is_ignored_name(self, expression: expr | Expr | None) -> bool:
    top_expression = expression.value if isinstance(expression, Expr) else expression
    if isinstance(top_expression, Attribute) and isinstance(top_expression.value, Name):
        name = top_expression.value.id
    elif isinstance(top_expression, Name):
        name = top_expression.id
    else:
        return False
    memo: TransformMemo | None = self
    while memo is not None:
        if name in memo.ignored_names:
            return True
        memo = memo.parent
    return False
from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def swap_all(coercions: BinaryCoercions) -> BinaryCoercions:
    return {**coercions, **{(b, a): swap_args(func) for (a, b), func in coercions.items()}}
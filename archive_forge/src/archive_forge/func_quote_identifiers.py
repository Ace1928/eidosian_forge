from __future__ import annotations
import itertools
import typing as t
from sqlglot import alias, exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import OptimizeError
from sqlglot.helper import seq_get, SingleValuedMapping
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.optimizer.scope import Scope, build_scope, traverse_scope, walk_in_scope
from sqlglot.optimizer.simplify import simplify_parens
from sqlglot.schema import Schema, ensure_schema
def quote_identifiers(expression: E, dialect: DialectType=None, identify: bool=True) -> E:
    """Makes sure all identifiers that need to be quoted are quoted."""
    return expression.transform(Dialect.get_or_raise(dialect).quote_identifier, identify=identify, copy=False)
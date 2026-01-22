from __future__ import annotations
import itertools
import typing as t
from sqlglot import alias, exp
from sqlglot.dialects.dialect import DialectType
from sqlglot.helper import csv_reader, name_sequence
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema

    Rewrite sqlglot AST to have fully qualified tables. Join constructs such as
    (t1 JOIN t2) AS t will be expanded into (SELECT * FROM t1 AS t1, t2 AS t2) AS t.

    Examples:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one("SELECT 1 FROM tbl")
        >>> qualify_tables(expression, db="db").sql()
        'SELECT 1 FROM db.tbl AS tbl'
        >>>
        >>> expression = sqlglot.parse_one("SELECT 1 FROM (t1 JOIN t2) AS t")
        >>> qualify_tables(expression).sql()
        'SELECT 1 FROM (SELECT * FROM t1 AS t1, t2 AS t2) AS t'

    Args:
        expression: Expression to qualify
        db: Database name
        catalog: Catalog name
        schema: A schema to populate
        dialect: The dialect to parse catalog and schema into.

    Returns:
        The qualified expression.
    
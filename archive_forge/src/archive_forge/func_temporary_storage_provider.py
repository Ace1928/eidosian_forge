from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.helper import seq_get
from sqlglot.transforms import (
def temporary_storage_provider(expression: exp.Expression) -> exp.Expression:
    provider = exp.FileFormatProperty(this=exp.Literal.string('parquet'))
    expression.args['properties'].append('expressions', provider)
    return expression
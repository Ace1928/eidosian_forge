from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.postgres import Postgres
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def datatype_sql(self, expression: exp.DataType) -> str:
    """
            Redshift converts the `TEXT` data type to `VARCHAR(255)` by default when people more generally mean
            VARCHAR of max length which is `VARCHAR(max)` in Redshift. Therefore if we get a `TEXT` data type
            without precision we convert it to `VARCHAR(max)` and if it does have precision then we just convert
            `TEXT` to `VARCHAR`.
            """
    if expression.is_type('text'):
        expression.set('this', exp.DataType.Type.VARCHAR)
        precision = expression.args.get('expressions')
        if not precision:
            expression.append('expressions', exp.var('MAX'))
    return super().datatype_sql(expression)
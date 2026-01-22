from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def struct_sql(self, expression: exp.Struct) -> str:
    from sqlglot.optimizer.annotate_types import annotate_types
    expression = annotate_types(expression)
    values: t.List[str] = []
    schema: t.List[str] = []
    unknown_type = False
    for e in expression.expressions:
        if isinstance(e, exp.PropertyEQ):
            if e.type and e.type.is_type(exp.DataType.Type.UNKNOWN):
                unknown_type = True
            else:
                schema.append(f'{self.sql(e, 'this')} {self.sql(e.type)}')
            values.append(self.sql(e, 'expression'))
        else:
            values.append(self.sql(e))
    size = len(expression.expressions)
    if not size or len(schema) != size:
        if unknown_type:
            self.unsupported('Cannot convert untyped key-value definitions (try annotate_types).')
        return self.func('ROW', *values)
    return f'CAST(ROW({', '.join(values)}) AS ROW({', '.join(schema)}))'
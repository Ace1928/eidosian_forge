from typing import Any, Optional
import pyarrow as pa
from fugue.column.expressions import (
from triad import Schema
def is_agg(column: Any) -> bool:
    """Check if a column contains aggregation operation

    :param col: the column to check
    :return: whether the column is :class:`~fugue.column.expressions.ColumnExpr`
      and contains aggregation operations

    .. admonition:: New Since
        :class: hint

        **0.6.0**

    .. admonition:: Examples

        .. code-block:: python

            import fugue.column.functions as f

            assert not f.is_agg(1)
            assert not f.is_agg(col("a"))
            assert not f.is_agg(col("a")+lit(1))

            assert f.is_agg(f.max(col("a")))
            assert f.is_agg(-f.max(col("a")))
            assert f.is_agg(f.max(col("a")+1))
            assert f.is_agg(f.max(col("a"))+f.min(col("a"))))
    """
    if isinstance(column, _UnaryAggFuncExpr):
        return True
    if isinstance(column, _FuncExpr):
        return any((is_agg(x) for x in column.args)) or any((is_agg(x) for x in column.kwargs.values()))
    return False
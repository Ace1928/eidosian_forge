from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Sequence, Union
from .core.property.vectorization import Expr, Field
from .models.expressions import CumSum, Stack
from .models.mappers import (
from .models.transforms import Dodge, Jitter
def factor_hatch(field_name: str, patterns: Sequence[str], factors: Factors, start: float=0, end: float | None=None) -> Field:
    """ Create a ``DataSpec`` dict that applies a client-side
    ``CategoricalPatternMapper`` transformation to a ``ColumnDataSource``
    column.

    Args:
        field_name (str) : a field name to configure ``DataSpec`` with

        patterns (seq[string]) : a list of hatch patterns to use to map to

        factors (seq) : a sequences of categorical factors corresponding to
            the palette

        start (int, optional) : a start slice index to apply when the column
            data has factors with multiple levels. (default: 0)

        end (int, optional) : an end slice index to apply when the column
            data has factors with multiple levels. (default: None)

    Returns:
        Field

    Added in version 1.1.1

    """
    return Field(field_name, CategoricalPatternMapper(patterns=patterns, factors=factors, start=start, end=end))
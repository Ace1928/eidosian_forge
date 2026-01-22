from __future__ import annotations
import logging # isort:skip
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from ..core.properties import Datetime
from ..core.property.singletons import Intrinsic
from ..models import (
def get_range(range_input: Range | tuple[float, float] | Sequence[str] | pd.Series[Any] | GroupBy | None) -> Range:
    import pandas as pd
    from pandas.core.groupby import GroupBy
    if range_input is None:
        return DataRange1d()
    if isinstance(range_input, GroupBy):
        return FactorRange(factors=sorted(list(range_input.groups.keys())))
    if isinstance(range_input, Range):
        return range_input
    if isinstance(range_input, pd.Series):
        range_input = range_input.values
    if isinstance(range_input, (Sequence, np.ndarray)):
        if all((isinstance(x, str) for x in range_input)):
            return FactorRange(factors=list(range_input))
        if len(range_input) == 2:
            try:
                start, end = range_input
                if start is None:
                    start = Intrinsic
                if end is None:
                    end = Intrinsic
                return Range1d(start=start, end=end)
            except ValueError:
                pass
    raise ValueError(f"Unrecognized range input: '{range_input}'")
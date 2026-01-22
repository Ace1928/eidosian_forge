from typing import Iterator, Optional, Tuple
import numpy as np
import pandas
from pandas._typing import AggFuncType, AggFuncTypeBase, AggFuncTypeDict, IndexLabel
from pandas.util._decorators import doc
from modin.utils import func_from_deprecated_location, hashable
from_pandas = func_from_deprecated_location(
from_arrow = func_from_deprecated_location(
from_dataframe = func_from_deprecated_location(
from_non_pandas = func_from_deprecated_location(
def walk_aggregation_dict(agg_dict: AggFuncTypeDict) -> Iterator[Tuple[IndexLabel, AggFuncTypeBase, Optional[str], bool]]:
    """
    Walk over an aggregation dictionary.

    Parameters
    ----------
    agg_dict : AggFuncTypeDict

    Yields
    ------
    (col: IndexLabel, func: AggFuncTypeBase, func_name: Optional[str], col_renaming_required: bool)
        Yield an aggregation function with its metadata:
            - `col`: column name to apply the function.
            - `func`: aggregation function to apply to the column.
            - `func_name`: custom function name that was specified in the dict.
            - `col_renaming_required`: whether it's required to rename the
                `col` into ``(col, func_name)``.
    """
    for key, value in agg_dict.items():
        yield from _walk_aggregation_func(key, value)
from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
def select_columns_by_name(self, names: list[str]) -> PandasDataFrameXchg:
    if not isinstance(names, abc.Sequence):
        raise ValueError('`names` is not a sequence')
    if not isinstance(names, list):
        names = list(names)
    return PandasDataFrameXchg(self._df.loc[:, names], allow_copy=self._allow_copy)
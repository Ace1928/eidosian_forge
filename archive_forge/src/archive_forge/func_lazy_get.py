from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
def lazy_get(self, ids: list, numeric_index: bool=False) -> 'ModinDtypes':
    """
        Get new ``ModinDtypes`` for a subset of columns without triggering any computations.

        Parameters
        ----------
        ids : list of index labels or positional indexers
            Columns for the subset.
        numeric_index : bool, default: False
            Whether `ids` are positional indixes or column labels to take.

        Returns
        -------
        ModinDtypes
            ``ModinDtypes`` that describes dtypes for columns specified in `ids`.
        """
    if isinstance(self._value, DtypesDescriptor):
        res = self._value.lazy_get(ids, numeric_index)
        return ModinDtypes(res)
    elif callable(self._value):
        new_self = self.copy()
        old_value = new_self._value
        new_self._value = lambda: old_value().iloc[ids] if numeric_index else old_value()[ids]
        return new_self
    ErrorMessage.catch_bugs_and_request_email(failure_condition=not self.is_materialized)
    return ModinDtypes(self._value.iloc[ids] if numeric_index else self._value[ids])
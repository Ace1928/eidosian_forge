from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
def maybe_specify_new_frame_ref(self, new_parent: 'PandasDataframe') -> 'ModinDtypes':
    """
        Set a new parent for the stored value if needed.

        Parameters
        ----------
        new_parent : PandasDataframe

        Returns
        -------
        ModinDtypes
            A copy of ``ModinDtypes`` with updated parent.
        """
    new_self = self.copy()
    if new_self.is_materialized:
        LazyProxyCategoricalDtype.update_dtypes(new_self._value, new_parent)
        return new_self
    if isinstance(self._value, DtypesDescriptor):
        new_self._value.update_parent(new_parent)
        return new_self
    return new_self
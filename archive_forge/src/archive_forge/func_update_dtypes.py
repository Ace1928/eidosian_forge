from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
@staticmethod
def update_dtypes(dtypes, new_parent):
    """
        Update a parent for categorical proxies in a dtype object.

        Parameters
        ----------
        dtypes : dict-like
            A dict-like object describing dtypes. The method will walk through every dtype
            an update parents for categorical proxies inplace.
        new_parent : object
        """
    for key, value in dtypes.items():
        if isinstance(value, LazyProxyCategoricalDtype):
            dtypes[key] = value._update_proxy(new_parent, column_name=key)
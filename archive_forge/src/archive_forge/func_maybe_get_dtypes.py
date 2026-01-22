import functools
import uuid
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.indexes.api import ensure_index
def maybe_get_dtypes(self):
    """
        Get index dtypes if available.

        Returns
        -------
        pandas.Series or None
        """
    if self._dtypes is not None:
        return self._dtypes
    if self.is_materialized:
        self._dtypes = self._value.dtypes if isinstance(self._value, pandas.MultiIndex) else pandas.Series([self._value.dtype], index=[self._value.name])
        return self._dtypes
    return None
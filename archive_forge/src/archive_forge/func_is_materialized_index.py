import functools
import uuid
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.indexes.api import ensure_index
@classmethod
def is_materialized_index(cls, index) -> bool:
    """
        Check if the passed object represents a materialized index.

        Parameters
        ----------
        index : object
            An object to check.

        Returns
        -------
        bool
        """
    from modin.pandas.indexing import is_range_like
    if isinstance(index, cls):
        index = index._value
    return is_list_like(index) or is_range_like(index) or isinstance(index, slice)
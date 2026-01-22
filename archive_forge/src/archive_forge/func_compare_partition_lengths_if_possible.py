import functools
import uuid
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.indexes.api import ensure_index
def compare_partition_lengths_if_possible(self, other: 'ModinIndex'):
    """
        Compare the partition lengths cache for the index being stored if possible.

        The ``ModinIndex`` object may sometimes store the information about partition
        lengths along the axis the index belongs to. If both `self` and `other` have
        this information or it can be inferred from them, the method returns
        a boolean - the result of the comparison, otherwise it returns ``None``
        as an indication that the comparison cannot be made.

        Parameters
        ----------
        other : ModinIndex

        Returns
        -------
        bool or None
            The result of the comparison if both `self` and `other` contain
            the lengths data, ``None`` otherwise.
        """
    if self._lengths_id == other._lengths_id:
        return True
    can_extract_lengths_from_self = self._lengths_cache is not None or callable(self._value)
    can_extract_lengths_from_other = other._lengths_cache is not None or callable(other._value)
    if can_extract_lengths_from_self and can_extract_lengths_from_other:
        return self.get(return_lengths=True)[1] == other.get(return_lengths=True)[1]
    return None
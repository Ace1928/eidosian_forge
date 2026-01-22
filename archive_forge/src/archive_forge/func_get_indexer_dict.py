from __future__ import annotations
from collections import defaultdict
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import unique_label_indices
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.construction import extract_array
def get_indexer_dict(label_list: list[np.ndarray], keys: list[Index]) -> dict[Hashable, npt.NDArray[np.intp]]:
    """
    Returns
    -------
    dict:
        Labels mapped to indexers.
    """
    shape = tuple((len(x) for x in keys))
    group_index = get_group_index(label_list, shape, sort=True, xnull=True)
    if np.all(group_index == -1):
        return {}
    ngroups = (group_index.size and group_index.max()) + 1 if is_int64_overflow_possible(shape) else np.prod(shape, dtype='i8')
    sorter = get_group_index_sorter(group_index, ngroups)
    sorted_labels = [lab.take(sorter) for lab in label_list]
    group_index = group_index.take(sorter)
    return lib.indices_fast(sorter, group_index, keys, sorted_labels)
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.errors import MergeError
from modin.core.dataframe.base.dataframe.utils import join_columns
from modin.core.dataframe.pandas.metadata import ModinDtypes
from .utils import merge_partitioning
def should_keep_index(left, right):
    keep_index = False
    if left_on is not None and right_on is not None:
        keep_index = any((o in left.index.names and o in right_on and (o in right.index.names) for o in left_on))
    elif on is not None:
        keep_index = any((o in left.index.names and o in right.index.names for o in on))
    return keep_index
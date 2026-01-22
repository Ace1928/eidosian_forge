import abc
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._libs.tslibs import to_offset
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype
from pandas.core.resample import _get_timestamp_range_edges
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
def lazy_metadata_decorator(apply_axis=None, axis_arg=-1, transpose=False):
    """
    Lazily propagate metadata for the ``PandasDataframe``.

    This decorator first adds the minimum required reindexing operations
    to each partition's queue of functions to be lazily applied for
    each PandasDataframe in the arguments by applying the function
    run_f_on_minimally_updated_metadata. The decorator also sets the
    flags for deferred metadata synchronization on the function result
    if necessary.

    Parameters
    ----------
    apply_axis : str, default: None
        The axes on which to apply the reindexing operations to the `self._partitions` lazily.
        Case None: No lazy metadata propagation.
        Case "both": Add reindexing operations on both axes to partition queue.
        Case "opposite": Add reindexing operations complementary to given axis.
        Case "rows": Add reindexing operations on row axis to partition queue.
    axis_arg : int, default: -1
        The index or column axis.
    transpose : bool, default: False
        Boolean for if a transpose operation is being used.

    Returns
    -------
    Wrapped Function.
    """

    def decorator(f):
        from functools import wraps

        @wraps(f)
        def run_f_on_minimally_updated_metadata(self, *args, **kwargs):
            from .dataframe import PandasDataframe
            for obj in [self] + [o for o in args if isinstance(o, PandasDataframe)] + [v for v in kwargs.values() if isinstance(v, PandasDataframe)] + [d for o in args if isinstance(o, list) for d in o if isinstance(d, PandasDataframe)] + [d for _, o in kwargs.items() if isinstance(o, list) for d in o if isinstance(d, PandasDataframe)]:
                if apply_axis == 'both':
                    if obj._deferred_index and obj._deferred_column:
                        obj._propagate_index_objs(axis=None)
                    elif obj._deferred_index:
                        obj._propagate_index_objs(axis=0)
                    elif obj._deferred_column:
                        obj._propagate_index_objs(axis=1)
                elif apply_axis == 'opposite':
                    if 'axis' not in kwargs:
                        axis = args[axis_arg]
                    else:
                        axis = kwargs['axis']
                    if axis == 0 and obj._deferred_column:
                        obj._propagate_index_objs(axis=1)
                    elif axis == 1 and obj._deferred_index:
                        obj._propagate_index_objs(axis=0)
                elif apply_axis == 'rows':
                    obj._propagate_index_objs(axis=0)
            result = f(self, *args, **kwargs)
            if apply_axis is None and (not transpose):
                result._deferred_index = self._deferred_index
                result._deferred_column = self._deferred_column
            elif apply_axis is None and transpose:
                result._deferred_index = self._deferred_column
                result._deferred_column = self._deferred_index
            elif apply_axis == 'opposite':
                if axis == 0:
                    result._deferred_index = self._deferred_index
                else:
                    result._deferred_column = self._deferred_column
            elif apply_axis == 'rows':
                result._deferred_column = self._deferred_column
            return result
        return run_f_on_minimally_updated_metadata
    return decorator
from __future__ import annotations
import datetime
import warnings
from numbers import Integral
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.core.window import Rolling as pd_Rolling
from dask.array.core import normalize_arg
from dask.base import tokenize
from dask.blockwise import BlockwiseDepDict
from dask.dataframe import methods
from dask.dataframe._compat import check_axis_keyword_deprecation
from dask.dataframe.core import (
from dask.dataframe.io import from_pandas
from dask.dataframe.multi import _maybe_align_partitions
from dask.dataframe.utils import (
from dask.delayed import unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import M, apply, derived_from, funcname, has_keyword
@insert_meta_param_description
def map_overlap(func, df, before, after, *args, meta=no_default, enforce_metadata=True, transform_divisions=True, align_dataframes=True, **kwargs):
    """Apply a function to each partition, sharing rows with adjacent partitions.

    Parameters
    ----------
    func : function
        The function applied to each partition. If this function accepts
        the special ``partition_info`` keyword argument, it will receive
        information on the partition's relative location within the
        dataframe.
    df: dd.DataFrame, dd.Series
    args, kwargs :
        Positional and keyword arguments to pass to the function.
        Positional arguments are computed on a per-partition basis, while
        keyword arguments are shared across all partitions. The partition
        itself will be the first positional argument, with all other
        arguments passed *after*. Arguments can be ``Scalar``, ``Delayed``,
        or regular Python objects. DataFrame-like args (both dask and
        pandas) will be repartitioned to align (if necessary) before
        applying the function; see ``align_dataframes`` to control this
        behavior.
    enforce_metadata : bool, default True
        Whether to enforce at runtime that the structure of the DataFrame
        produced by ``func`` actually matches the structure of ``meta``.
        This will rename and reorder columns for each partition,
        and will raise an error if this doesn't work,
        but it won't raise if dtypes don't match.
    before : int, timedelta or string timedelta
        The rows to prepend to partition ``i`` from the end of
        partition ``i - 1``.
    after : int, timedelta or string timedelta
        The rows to append to partition ``i`` from the beginning
        of partition ``i + 1``.
    transform_divisions : bool, default True
        Whether to apply the function onto the divisions and apply those
        transformed divisions to the output.
    align_dataframes : bool, default True
        Whether to repartition DataFrame- or Series-like args
        (both dask and pandas) so their divisions align before applying
        the function. This requires all inputs to have known divisions.
        Single-partition inputs will be split into multiple partitions.

        If False, all inputs must have either the same number of partitions
        or a single partition. Single-partition inputs will be broadcast to
        every partition of multi-partition inputs.
    $META

    See Also
    --------
    dd.DataFrame.map_overlap
    """
    df = from_pandas(df, 1) if (is_series_like(df) or is_dataframe_like(df)) and (not is_dask_collection(df)) else df
    args = (df,) + args
    if isinstance(before, str):
        before = pd.to_timedelta(before)
    if isinstance(after, str):
        after = pd.to_timedelta(after)
    if isinstance(before, datetime.timedelta) or isinstance(after, datetime.timedelta):
        if not is_datetime64_any_dtype(df.index._meta_nonempty.inferred_type):
            raise TypeError('Must have a `DatetimeIndex` when using string offset for `before` and `after`')
    elif not (isinstance(before, Integral) and before >= 0 and isinstance(after, Integral) and (after >= 0)):
        raise ValueError('before and after must be positive integers')
    name = kwargs.pop('token', None)
    parent_meta = kwargs.pop('parent_meta', None)
    assert callable(func)
    if name is not None:
        token = tokenize(meta, before, after, *args, **kwargs)
    else:
        name = 'overlap-' + funcname(func)
        token = tokenize(func, meta, before, after, *args, **kwargs)
    name = f'{name}-{token}'
    if align_dataframes:
        args = _maybe_from_pandas(args)
        try:
            args = _maybe_align_partitions(args)
        except ValueError as e:
            raise ValueError(f"{e}. If you don't want the partitions to be aligned, and are calling `map_overlap` directly, pass `align_dataframes=False`.") from e
    dfs = [df for df in args if isinstance(df, _Frame)]
    meta = _get_meta_map_partitions(args, dfs, func, kwargs, meta, parent_meta)
    if all((isinstance(arg, Scalar) for arg in args)):
        layer = {(name, 0): (apply, func, (tuple, [(arg._name, 0) for arg in args]), kwargs)}
        graph = HighLevelGraph.from_collections(name, layer, dependencies=args)
        return Scalar(graph, name, meta)
    args2 = []
    dependencies = []
    divisions = _get_divisions_map_partitions(align_dataframes, transform_divisions, dfs, func, args, kwargs)

    def _handle_frame_argument(arg):
        dsk = {}
        prevs_parts_dsk, prevs = _get_previous_partitions(arg, before)
        dsk.update(prevs_parts_dsk)
        nexts_parts_dsk, nexts = _get_nexts_partitions(arg, after)
        dsk.update(nexts_parts_dsk)
        name_a = 'overlap-concat-' + tokenize(arg)
        for i, (prev, current, next) in enumerate(zip(prevs, arg.__dask_keys__(), nexts)):
            key = (name_a, i)
            dsk[key] = (_combined_parts, prev, current, next, before, after)
        graph = HighLevelGraph.from_collections(name_a, dsk, dependencies=[arg])
        return new_dd_object(graph, name_a, meta, divisions)
    for arg in args:
        if isinstance(arg, _Frame):
            arg = _handle_frame_argument(arg)
            args2.append(arg)
            dependencies.append(arg)
            continue
        arg = normalize_arg(arg)
        arg2, collections = unpack_collections(arg)
        if collections:
            args2.append(arg2)
            dependencies.extend(collections)
        else:
            args2.append(arg)
    kwargs3 = {}
    simple = True
    for k, v in kwargs.items():
        v = normalize_arg(v)
        v, collections = unpack_collections(v)
        dependencies.extend(collections)
        kwargs3[k] = v
        if collections:
            simple = False
    if has_keyword(func, 'partition_info'):
        partition_info = {(i,): {'number': i, 'division': division} for i, division in enumerate(divisions[:-1])}
        args2.insert(0, BlockwiseDepDict(partition_info))
        orig_func = func

        def func(partition_info, *args, **kwargs):
            return orig_func(*args, **kwargs, partition_info=partition_info)
    if enforce_metadata:
        dsk = partitionwise_graph(apply_and_enforce, name, func, before, after, *args2, dependencies=dependencies, _func=overlap_chunk, _meta=meta, **kwargs3)
    else:
        kwargs4 = kwargs if simple else kwargs3
        dsk = partitionwise_graph(overlap_chunk, name, func, before, after, *args2, **kwargs4, dependencies=dependencies)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)
    return new_dd_object(graph, name, meta, divisions)
from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, union_categoricals
from dask.array.core import Array
from dask.array.dispatch import percentile_lookup
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
from dask.dataframe._compat import PANDAS_GE_220, is_any_real_numeric_dtype
from dask.dataframe.core import DataFrame, Index, Scalar, Series, _Frame
from dask.dataframe.dispatch import (
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.dataframe.utils import (
from dask.sizeof import SimpleSizeof, sizeof
from dask.utils import is_arraylike, is_series_like, typename
@concat_dispatch.register((pd.DataFrame, pd.Series, pd.Index))
def concat_pandas(dfs, axis=0, join='outer', uniform=False, filter_warning=True, ignore_index=False, **kwargs):
    ignore_order = kwargs.pop('ignore_order', False)
    if axis == 1:
        return pd.concat(dfs, axis=axis, join=join, **kwargs)
    if isinstance(dfs[0], pd.Index):
        if isinstance(dfs[0], pd.CategoricalIndex):
            for i in range(1, len(dfs)):
                if not isinstance(dfs[i], pd.CategoricalIndex):
                    dfs[i] = dfs[i].astype('category')
            return pd.CategoricalIndex(union_categoricals(dfs, ignore_order=ignore_order), name=dfs[0].name)
        elif isinstance(dfs[0], pd.MultiIndex):
            first, rest = (dfs[0], dfs[1:])
            if all((isinstance(o, pd.MultiIndex) and o.nlevels >= first.nlevels for o in rest)):
                arrays = [concat([i._get_level_values(n) for i in dfs]) for n in range(first.nlevels)]
                return pd.MultiIndex.from_arrays(arrays, names=first.names)
            to_concat = (first.values,) + tuple((k._values for k in rest))
            new_tuples = np.concatenate(to_concat)
            try:
                return pd.MultiIndex.from_tuples(new_tuples, names=first.names)
            except Exception:
                return pd.Index(new_tuples)
        return dfs[0].append(dfs[1:])
    dfs0_index = dfs[0].index
    has_categoricalindex = isinstance(dfs0_index, pd.CategoricalIndex) or (isinstance(dfs0_index, pd.MultiIndex) and any((isinstance(i, pd.CategoricalIndex) for i in dfs0_index.levels)))
    if has_categoricalindex:
        dfs2 = [df.reset_index(drop=True) for df in dfs]
        ind = concat([df.index for df in dfs])
    else:
        dfs2 = dfs
        ind = None
    if isinstance(dfs2[0], pd.DataFrame) if uniform else any((isinstance(df, pd.DataFrame) for df in dfs2)):
        if uniform or PANDAS_GE_220:
            dfs3 = dfs2
            cat_mask = dfs2[0].dtypes == 'category'
        else:
            dfs3 = [df if isinstance(df, pd.DataFrame) else df.to_frame().rename(columns={df.name: 0}) for df in dfs2]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                if filter_warning:
                    warnings.simplefilter('ignore', FutureWarning)
                cat_mask = pd.concat([(df.dtypes == 'category').to_frame().T for df in dfs3], join=join, **kwargs).any()
        if isinstance(cat_mask, pd.Series) and cat_mask.any():
            not_cat = cat_mask[~cat_mask].index
            out = pd.concat([df[df.columns.intersection(not_cat)] for df in dfs3], join=join, **kwargs)
            temp_ind = out.index
            for col in cat_mask.index.difference(not_cat):
                for df in dfs3:
                    sample = df.get(col)
                    if sample is not None:
                        break
                parts = []
                for df in dfs3:
                    if col in df.columns:
                        parts.append(df[col])
                    else:
                        codes = np.full(len(df), -1, dtype='i8')
                        data = pd.Categorical.from_codes(codes, sample.cat.categories, sample.cat.ordered)
                        parts.append(data)
                out[col] = union_categoricals(parts, ignore_order=ignore_order)
                if not len(temp_ind):
                    out.index = temp_ind
            out = out.reindex(columns=cat_mask.index)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                if filter_warning:
                    warnings.simplefilter('ignore', FutureWarning)
                out = pd.concat(dfs3, join=join, sort=False)
    else:
        if isinstance(dfs2[0].dtype, pd.CategoricalDtype):
            if ind is None:
                ind = concat([df.index for df in dfs2])
            return pd.Series(union_categoricals(dfs2, ignore_order=ignore_order), index=ind, name=dfs2[0].name)
        with warnings.catch_warnings():
            if filter_warning:
                warnings.simplefilter('ignore', FutureWarning)
            out = pd.concat(dfs2, join=join, **kwargs)
    if ind is not None:
        out.index = ind
    return out
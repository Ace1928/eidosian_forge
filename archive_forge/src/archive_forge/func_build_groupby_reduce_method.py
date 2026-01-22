import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@classmethod
def build_groupby_reduce_method(cls, agg_func):
    """
        Build function for `QueryCompiler.groupby_*` that can be executed as default-to-pandas.

        Parameters
        ----------
        agg_func : callable or str
            Default aggregation function. If aggregation function is not specified
            via groupby arguments, then `agg_func` function is used.

        Returns
        -------
        callable
            Function that executes groupby aggregation.
        """

    def fn(df, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False, **kwargs):
        """Group DataFrame and apply aggregation function to each group."""
        if not isinstance(by, (pandas.Series, pandas.DataFrame)):
            by = cls.validate_by(by)
            grp = cls._call_groupby(df, by, axis=axis, **groupby_kwargs)
            grp_agg_func = cls.get_func(agg_func, **kwargs)
            return grp_agg_func(grp, *agg_args, **agg_kwargs)
        if isinstance(by, pandas.DataFrame):
            by = by.squeeze(axis=1)
        if drop and isinstance(by, pandas.Series) and (by.name in df) and df[by.name].equals(by):
            by = [by.name]
        if isinstance(by, pandas.DataFrame):
            df = pandas.concat([df] + [by[[o for o in by if o not in df]]], axis=1)
            by = list(by.columns)
        groupby_kwargs = groupby_kwargs.copy()
        as_index = groupby_kwargs.pop('as_index', True)
        groupby_kwargs['as_index'] = True
        grp = cls._call_groupby(df, by, axis=axis, **groupby_kwargs)
        func = cls.get_func(agg_func, **kwargs)
        result = func(grp, *agg_args, **agg_kwargs)
        method = kwargs.get('method')
        if isinstance(result, pandas.Series):
            result = result.to_frame(MODIN_UNNAMED_SERIES_LABEL if result.name is None else result.name)
        if not as_index:
            if isinstance(by, pandas.Series):
                internal_by = (by.name,) if drop or method == 'size' else tuple()
            else:
                internal_by = by
            cls.handle_as_index_for_dataframe(result, internal_by, by_cols_dtypes=df.index.dtypes.values if isinstance(df.index, pandas.MultiIndex) else (df.index.dtype,), by_length=len(by), drop=drop, method=method, inplace=True)
        if result.index.name == MODIN_UNNAMED_SERIES_LABEL:
            result.index.name = None
        return result
    return fn
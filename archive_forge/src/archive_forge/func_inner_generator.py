import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet
from ._monitor import TMonitor
from .utils import (
def inner_generator(df_function='apply'):

    def inner(df, func, *args, **kwargs):
        """
                Parameters
                ----------
                df  : (DataFrame|Series)[GroupBy]
                    Data (may be grouped).
                func  : function
                    To be applied on the (grouped) data.
                **kwargs  : optional
                    Transmitted to `df.apply()`.
                """
        total = tqdm_kwargs.pop('total', getattr(df, 'ngroups', None))
        if total is None:
            if df_function == 'applymap':
                total = df.size
            elif isinstance(df, Series):
                total = len(df)
            elif _Rolling_and_Expanding is None or not isinstance(df, _Rolling_and_Expanding):
                axis = kwargs.get('axis', 0)
                if axis == 'index':
                    axis = 0
                elif axis == 'columns':
                    axis = 1
                total = df.size // df.shape[axis]
        if deprecated_t[0] is not None:
            t = deprecated_t[0]
            deprecated_t[0] = None
        else:
            t = cls(total=total, **tqdm_kwargs)
        if len(args) > 0:
            TqdmDeprecationWarning('Except func, normal arguments are intentionally' + ' not supported by' + ' `(DataFrame|Series|GroupBy).progress_apply`.' + ' Use keyword arguments instead.', fp_write=getattr(t.fp, 'write', sys.stderr.write))
        try:
            from pandas.core.common import is_builtin_func
        except ImportError:
            is_builtin_func = df._is_builtin_func
        try:
            func = is_builtin_func(func)
        except TypeError:
            pass

        def wrapper(*args, **kwargs):
            t.update(n=1 if not t.total or t.n < t.total else 0)
            return func(*args, **kwargs)
        try:
            return getattr(df, df_function)(wrapper, **kwargs)
        finally:
            t.close()
    return inner
from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
def inject_reduce_methods(cls):
    methods = [(name, getattr(duck_array_ops, f'array_{name}'), False) for name in REDUCE_METHODS] + [(name, getattr(duck_array_ops, name), True) for name in NAN_REDUCE_METHODS] + [('count', duck_array_ops.count, False)]
    for name, f, include_skipna in methods:
        numeric_only = getattr(f, 'numeric_only', False)
        available_min_count = getattr(f, 'available_min_count', False)
        skip_na_docs = _SKIPNA_DOCSTRING if include_skipna else ''
        min_count_docs = _MINCOUNT_DOCSTRING if available_min_count else ''
        func = cls._reduce_method(f, include_skipna, numeric_only)
        func.__name__ = name
        func.__doc__ = _REDUCE_DOCSTRING_TEMPLATE.format(name=name, cls=cls.__name__, extra_args=cls._reduce_extra_args_docstring.format(name=name), skip_na_docs=skip_na_docs, min_count_docs=min_count_docs)
        setattr(cls, name, func)
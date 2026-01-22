from __future__ import annotations
from itertools import count
import logging
from typing import TYPE_CHECKING
from toolz import unique, concat, pluck, get, memoize
from numba import literal_unroll
import numpy as np
import xarray as xr
from .antialias import AntialiasCombination
from .reductions import SpecialColumn, UsesCudaMutex, by, category_codes, summary
from .utils import (isnull, ngjit,
def make_finalize(bases, agg, schema, cuda, partitioned):
    arg_lk = dict(((k, v) for v, k in enumerate(bases)))
    if isinstance(agg, summary):
        calls = []
        for key, val in zip(agg.keys, agg.values):
            f = make_finalize(bases, val, schema, cuda, partitioned)
            try:
                bases = val._build_bases(cuda, partitioned)
            except AttributeError:
                pass
            inds = [arg_lk[b] for b in bases]
            calls.append((key, f, inds))

        def finalize(bases, cuda=False, **kwargs):
            data = {key: finalizer(get(inds, bases), cuda, **kwargs) for key, finalizer, inds in calls}
            name = agg.keys[0]
            attrs = {attr: data[name].attrs[attr] for attr in ('x_range', 'y_range')}
            return xr.Dataset(data, attrs=attrs)
        return finalize
    else:
        return agg._build_finalize(schema)
from __future__ import division, print_function, absolute_import
from petl.compat import next, string_types
from petl.util.base import iterpeek, ValuesView, Table
from petl.util.materialise import columns
def construct_dtype(flds, peek, dtype):
    import numpy as np
    if dtype is None:
        dtype = infer_dtype(peek)
    elif isinstance(dtype, string_types):
        typestrings = [s.strip() for s in dtype.split(',')]
        dtype = [(f, t) for f, t in zip(flds, typestrings)]
    elif isinstance(dtype, dict) and ('names' not in dtype or 'formats' not in dtype):
        cols = columns(peek)
        newdtype = {'names': [], 'formats': []}
        for f in flds:
            newdtype['names'].append(f)
            if f in dtype and isinstance(dtype[f], tuple):
                newdtype['formats'].append(dtype[f][0])
            elif f not in dtype:
                a = np.array(cols[f])
                newdtype['formats'].append(a.dtype)
            else:
                newdtype['formats'].append(dtype[f])
        dtype = newdtype
    return dtype
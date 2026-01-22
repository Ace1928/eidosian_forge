from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def sorted_by_keys(d):
    if 'keys2' in d:
        keys = tuple(zip(d['keys'], d['keys2']))
    else:
        keys = d['keys']
    sorted_keys = sorted(keys)
    sorted_d = {'keys': sorted(d['keys'])}
    for entry in d:
        if entry == 'keys':
            continue
        values = dict(zip(keys, d[entry]))
        for k in sorted_keys:
            sorted_d.setdefault(entry, []).append(values[k])
    return sorted_d
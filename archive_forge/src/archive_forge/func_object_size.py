from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
def object_size(*xs):
    if not xs:
        return 0
    ncells = sum((len(x) for x in xs))
    if not ncells:
        return 0
    unique_samples = {}
    for x in xs:
        sample = np.random.choice(x, size=100, replace=True)
        for i in sample.tolist():
            unique_samples[id(i)] = i
    nsamples = 100 * len(xs)
    sample_nbytes = sum((sizeof(i) for i in unique_samples.values()))
    if len(unique_samples) / nsamples > 0.5:
        return int(sample_nbytes * ncells / nsamples)
    else:
        return sample_nbytes
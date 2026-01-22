import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def sparse_random_normal(loc=0.0, scale=1.0, size=None, dtype=None, **kwargs):

    def rvs(nnz):
        return do('random.normal', loc, scale, (nnz,), dtype=dtype, like='numpy')
    return do('random', size, data_rvs=rvs, **kwargs, like='sparse')
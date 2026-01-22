import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
def merge_dimensions(dimensions_list):
    """
    Merges lists of fully or partially overlapping dimensions by
    merging their values.

    >>> from holoviews import Dimension
    >>> dim_list = [[Dimension('A', values=[1, 2, 3]), Dimension('B')],
    ...             [Dimension('A', values=[2, 3, 4])]]
    >>> dimensions = merge_dimensions(dim_list)
    >>> dimensions
    [Dimension('A'), Dimension('B')]
    >>> dimensions[0].values
    [1, 2, 3, 4]
    """
    dvalues = defaultdict(list)
    dimensions = []
    for dims in dimensions_list:
        for d in dims:
            dvalues[d.name].append(d.values)
            if d not in dimensions:
                dimensions.append(d)
    dvalues = {k: list(unique_iterator(itertools.chain(*vals))) for k, vals in dvalues.items()}
    return [d.clone(values=dvalues.get(d.name, [])) for d in dimensions]
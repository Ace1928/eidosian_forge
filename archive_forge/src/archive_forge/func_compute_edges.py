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
def compute_edges(edges):
    """
    Computes edges as midpoints of the bin centers.  The first and
    last boundaries are equidistant from the first and last midpoints
    respectively.
    """
    edges = np.asarray(edges)
    if edges.dtype.kind == 'i':
        edges = edges.astype('f')
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    boundaries = (2 * edges[0] - midpoints[0], 2 * edges[-1] - midpoints[-1])
    return np.concatenate([boundaries[:1], midpoints, boundaries[-1:]])
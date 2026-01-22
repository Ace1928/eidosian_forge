import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def get_path_fn(path_type):
    """Get the correct path finding function from str ``path_type``.
    """
    if path_type not in _PATH_OPTIONS:
        raise KeyError("Path optimizer '{}' not found, valid options are {}.".format(path_type, set(_PATH_OPTIONS.keys())))
    return _PATH_OPTIONS[path_type]
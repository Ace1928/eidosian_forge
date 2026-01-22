import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tree_map_tuple(f, tree, is_leaf):
    return tuple((tree_map(f, x, is_leaf) for x in tree))
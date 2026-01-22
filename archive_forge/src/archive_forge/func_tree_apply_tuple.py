import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tree_apply_tuple(f, tree, is_leaf):
    for x in tree:
        tree_apply(f, x, is_leaf)
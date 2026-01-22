import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tree_apply_dict(f, tree, is_leaf):
    for v in tree.values():
        tree_apply(f, v, is_leaf)
import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def is_leaf_placeholder(x):
    return x.__class__ is Leaf
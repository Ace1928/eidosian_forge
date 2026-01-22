import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def jax_to_numpy(x):
    return do('asarray', x, like='numpy')
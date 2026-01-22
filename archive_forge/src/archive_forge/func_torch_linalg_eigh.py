import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_linalg_eigh(x):
    return tuple(do('symeig', x, eigenvectors=True, like='torch'))
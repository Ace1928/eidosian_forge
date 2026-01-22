import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_linalg_eigvalsh(x):
    return do('symeig', x, eigenvectors=False, like='torch')[0]
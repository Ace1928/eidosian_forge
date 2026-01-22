import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_imag(x):
    try:
        if x.is_complex():
            return x.imag
    except AttributeError:
        pass
    return do('zeros_like', x, like='torch')
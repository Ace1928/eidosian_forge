import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_get_dtype_name(x):
    return _torch_get_dtype_name(x.dtype)
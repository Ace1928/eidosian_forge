import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
@size.register('torch')
def torch_size(x):
    return x.numel()
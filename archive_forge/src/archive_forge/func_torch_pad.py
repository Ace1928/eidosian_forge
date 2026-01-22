import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_pad(array, pad_width, mode='constant', constant_values=0):
    if mode != 'constant':
        raise NotImplementedError
    try:
        pad = tuple(itertools.chain.from_iterable(pad_width))[::-1]
        if len(pad) == 2:
            pad = pad * array.ndimension()
    except TypeError:
        pad = (pad_width,) * 2 * array.ndimension()
    return do('nn.functional.pad', array, pad=pad, mode=mode, value=constant_values, like='torch')
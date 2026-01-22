import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tensorflow_pad_wrap(tf_pad):

    def numpy_like(array, pad_width, mode='constant', constant_values=0):
        if mode != 'constant':
            raise NotImplementedError
        try:
            if len(pad_width) == 1:
                pad_width = pad_width * ndim(array)
        except TypeError:
            pad_width = ((pad_width, pad_width),) * ndim(array)
        return tf_pad(array, pad_width, mode='CONSTANT', constant_values=constant_values)
    return numpy_like
import importlib
import numpy
from . import object_arrays
from . import cupy as _cupy
from . import jax as _jax
from . import tensorflow as _tensorflow
from . import theano as _theano
from . import torch as _torch
def has_einsum(backend):
    """Check if ``{backend}.einsum`` exists, cache result for performance.
    """
    try:
        return _has_einsum[backend]
    except KeyError:
        try:
            get_func('einsum', backend)
            _has_einsum[backend] = True
        except AttributeError:
            _has_einsum[backend] = False
        return _has_einsum[backend]
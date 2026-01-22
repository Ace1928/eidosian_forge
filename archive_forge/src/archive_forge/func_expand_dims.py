import inspect
import tempfile
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any, ContextManager, Iterable, List, Tuple
import numpy as np
from packaging import version
from .import_utils import get_torch_version, is_flax_available, is_tf_available, is_torch_available, is_torch_fx_proxy
def expand_dims(array, axis):
    """
    Framework-agnostic version of `numpy.expand_dims` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    if is_numpy_array(array):
        return np.expand_dims(array, axis)
    elif is_torch_tensor(array):
        return array.unsqueeze(dim=axis)
    elif is_tf_tensor(array):
        import tensorflow as tf
        return tf.expand_dims(array, axis=axis)
    elif is_jax_tensor(array):
        return jnp.expand_dims(array, axis=axis)
    else:
        raise ValueError(f'Type not supported for expand_dims: {type(array)}.')
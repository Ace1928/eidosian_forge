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
def to_numpy(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a Numpy array.
    """
    framework_to_numpy = {'pt': lambda obj: obj.detach().cpu().numpy(), 'tf': lambda obj: obj.numpy(), 'jax': lambda obj: np.asarray(obj), 'np': lambda obj: obj}
    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_numpy[framework](obj)
    return obj
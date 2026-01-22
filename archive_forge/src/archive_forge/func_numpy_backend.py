from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
@staticmethod
@pytest.fixture()
def numpy_backend():
    kwargs = {'id_to_col': {1: 0}, 'param_to_size': {-1: 1, 2: 2}, 'param_to_col': {2: 0, -1: 2}, 'param_size_plus_one': 3, 'var_length': 2}
    backend = CanonBackend.get_backend(s.NUMPY_CANON_BACKEND, **kwargs)
    assert isinstance(backend, NumPyCanonBackend)
    return backend
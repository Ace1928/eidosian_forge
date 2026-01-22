from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_tensor_view_add_dicts(self, scipy_backend):
    view = scipy_backend.get_empty_view()
    one = sp.eye(1)
    two = sp.eye(1) * 2
    three = sp.eye(1) * 3
    assert view.add_dicts({}, {}) == {}
    assert view.add_dicts({'a': one}, {'a': two}) == {'a': three}
    assert view.add_dicts({'a': one}, {'b': two}) == {'a': one, 'b': two}
    assert view.add_dicts({'a': {'c': one}}, {'a': {'c': one}}) == {'a': {'c': two}}
    with pytest.raises(ValueError, match="Values must either be dicts or <class 'scipy.sparse."):
        view.add_dicts({'a': 1}, {'a': 2})
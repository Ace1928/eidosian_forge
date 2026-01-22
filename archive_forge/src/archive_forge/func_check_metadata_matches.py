import itertools
import unittest.mock
from contextlib import contextmanager
from typing import Iterator
import torch
import torch._C
import torch._ops
import torch.utils._python_dispatch
import torch.utils._pytree as pytree
def check_metadata_matches(n, r, desc):
    assert callable(desc)
    n_vals, n_spec = pytree.tree_flatten(n)
    r_vals, r_spec = pytree.tree_flatten(r)
    assert len(n_vals) == len(r_vals), f'{len(n_vals)} != {len(r_vals)}'
    for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
        if not isinstance(rv, torch.Tensor):
            continue
        check_tensor_metadata_matches(nv, rv, lambda: f'{desc()} output {i}')
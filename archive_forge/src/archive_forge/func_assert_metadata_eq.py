import contextlib
import warnings
import weakref
from typing import ContextManager, List, Optional, Tuple, TYPE_CHECKING
import torch
from torch._C._functorch import (
from torch._guards import Source
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from torch.utils.weak import WeakIdRef
import torch._prims_common as utils
def assert_metadata_eq(assert_eq, m1, m2, *, skip_symbolic=False):

    def go(m1, m2):
        assert_eq(m1.dtype, m2.dtype)
        if not skip_symbolic:
            assert_eq(m1.shape, m2.shape)
        assert_eq(m1.requires_grad, m2.requires_grad)
        assert_eq(m1.is_leaf, m2.is_leaf)
        assert_eq(m1.grad_fn is None, m2.grad_fn is None)
        assert_eq(m1.is_sparse, m2.is_sparse)
        assert_eq(m1.is_inference(), m2.is_inference())
        assert_eq(m1.is_conj(), m2.is_conj())
        assert_eq(m1.is_neg(), m2.is_neg())
        assert_eq(safe_grad(m1) is not None, safe_grad(m2) is not None)
        if safe_grad(m1) is not None:
            go(safe_grad(m1), safe_grad(m2))
        if m1.is_sparse:
            assert_eq(m1.dense_dim(), m2.dense_dim())
            assert_eq(m1.sparse_dim(), m2.sparse_dim())
            assert_eq(m1.is_coalesced(), m2.is_coalesced())
        else:
            if not skip_symbolic:
                assert_eq(m1.stride(), m2.stride())
                assert_eq(m1.storage_offset(), m2.storage_offset())
            assert_eq(m1._is_view(), m2._is_view())
            if m1._is_view():
                go(m1._base, m2._base)
    return go(m1, m2)
import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._higher_order_ops.triton_kernel_wrap import (
from torch._prims_common import (
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
from .utils import (
from .virtualized import ops, V
from . import kernel
import_submodule(kernel)
from . import quantized_lowerings
@register_lowering(aten.scatter_reduce_, type_promotion_kind=None)
def scatter_reduce_(self, dim: int, index, src, reduce, *, include_self: bool=True):
    assert reduce in {None, 'sum', 'prod', 'mean', 'amax', 'amin'}
    fallback_result = scatter_fallback('aten.scatter_reduce_', self, dim, index, src, reduce=reduce, include_self=include_self)
    if fallback_result:
        return fallback_result
    assert isinstance(self, TensorBox)
    assert 'int' in str(index.get_dtype())
    ndim = len(self.get_size())
    if ndim == 0:
        self = view(self, [1])
    if isinstance(src, TensorBox) and len(src.get_size()) == 0:
        src = view(src, [1])
    if isinstance(index, TensorBox) and len(index.get_size()) == 0:
        index = view(index, [1])
    dim = _validate_dim(self, dim)
    self.realize()
    index_loader = index.make_loader()
    src_loader = src.make_loader() if isinstance(src, TensorBox) else None

    def output_indexer(idx):
        shape = self.get_size()
        ndim = len(shape)
        indirect_idx = list(idx)
        indirect_idx[dim] = ops.indirect_indexing(index_loader(idx), 1 if ndim == 0 else shape[dim])
        return indirect_idx

    def fn(idx):
        if src_loader:
            return src_loader(idx)
        else:
            return ops.constant(src, self.get_dtype())

    def backend_reduce_str(reduce):
        if reduce == 'sum':
            return 'atomic_add'
        else:
            assert reduce is None
            return None
    if not include_self:
        zero_out = ir.Scatter(device=self.get_device(), dtype=self.get_dtype(), inner_fn=lambda index: ops.constant(0, self.get_dtype()), ranges=index.get_size(), output_indexer=output_indexer, scatter_mode=None)
        buffer = ir.ComputedBuffer(None, ir.MutationLayout(self), zero_out)
        buffer.name = V.graph.register_buffer(buffer)
    scatter = ir.Scatter(device=self.get_device(), dtype=self.get_dtype(), inner_fn=fn, ranges=index.get_size(), output_indexer=output_indexer, scatter_mode=backend_reduce_str(reduce))
    buffer = ir.ComputedBuffer(None, ir.MutationLayout(self), scatter)
    buffer.name = V.graph.register_buffer(buffer)
    if ndim == 0:
        self = view(self, [])
    return self
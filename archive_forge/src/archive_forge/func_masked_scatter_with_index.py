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
@register_lowering(inductor_prims.masked_scatter_with_index, type_promotion_kind=None, broadcast=False)
def masked_scatter_with_index(self, mask, source_idx, source):
    self_flat, mask_flat, source_flat = (view(x, (-1,)) for x in (self, mask, source))
    assert self.get_size() == mask.get_size()
    assert mask.get_dtype() in {torch.bool, torch.uint8}
    self_loader = self_flat.make_loader()
    mask_loader = mask_flat.make_loader()
    source_idx_loader = source_idx.make_loader()
    source_loader = source_flat.make_loader()
    source_numel = source.get_numel()

    def inner_fn(idx):
        self_val = self_loader(idx)
        mask_val = ops.to_dtype(mask_loader(idx), torch.bool)

        def load_source_val():
            source_idx_val = source_idx_loader(idx)
            i = ops.indirect_indexing(source_idx_val, source_numel)
            return source_loader([i])
        source_val = ops.masked(mask_val, load_source_val, 0)
        return ops.where(mask_val, source_val, self_val)
    result_flat = Pointwise.create(device=self.get_device(), dtype=self.get_dtype(), inner_fn=inner_fn, ranges=self_flat.get_size())
    return view(result_flat, self.get_size())
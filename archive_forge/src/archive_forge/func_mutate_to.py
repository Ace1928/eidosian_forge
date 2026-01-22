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
def mutate_to(changed, val, unsafe_alias=False):
    if isinstance(changed, TensorBox):
        changed_data = changed.data
    else:
        changed_data = changed
    if isinstance(val, TensorBox):
        val = val.data
    if not isinstance(val, ir.StorageBox):
        val = Pointwise.create(device=changed.get_device(), dtype=changed.get_dtype(), inner_fn=val.make_loader(), ranges=changed.get_size()).data
        assert isinstance(val, ir.StorageBox)
    if isinstance(changed_data, ir.StorageBox) and (not (changed_data.is_input_buffer() or isinstance(changed_data.data, ir.NopKernel))):
        val.realize()
        changed_data.data = val.data
        return changed
    ir.MutationLayout.realize_into(val, changed_data, unsafe_alias=unsafe_alias)
    return changed
from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import (
from torch.utils import _pytree as pytree
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
import torch.autograd.forward_ad as fwAD
from torch._subclasses.functional_tensor import FunctionalTensor
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes
from .apis import vmap
from torch._C._functorch import (
from torch._functorch.utils import exposed_in, argnums_t
def compute_jacobian_stacked():
    chunked_results = []
    for flat_basis_chunk in _chunked_standard_basis_for_(flat_output, flat_output_numels, chunk_size=chunk_size):
        if chunk_size == 1:
            for t in flat_basis_chunk:
                assert t.size(0) == 1
            flat_basis_chunk = tree_map(lambda t: torch.squeeze(t, 0), flat_basis_chunk)
        basis = tree_unflatten(flat_basis_chunk, output_spec)
        if chunk_size == 1:
            chunked_result = vjp_fn(basis)
        else:
            chunked_result = vmap(vjp_fn)(basis)
        flat_results = pytree.tree_leaves(chunked_result)
        if chunk_size == 1:
            flat_results = tree_map(lambda t: torch.unsqueeze(t, 0), flat_results)
        chunked_results.append(flat_results)
    if len(chunked_results) == 1:
        return chunked_results[0]
    flat_results = []
    for idx in range(len(flat_primals)):
        r = tuple((r_[idx] for r_ in chunked_results))
        flat_results.append(torch.cat(r, 0))
    return flat_results
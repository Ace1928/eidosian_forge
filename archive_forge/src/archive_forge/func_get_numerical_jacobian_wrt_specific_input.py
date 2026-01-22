import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def get_numerical_jacobian_wrt_specific_input(fn, input_idx, inputs, outputs, eps, input=None, is_forward_ad=False) -> Tuple[torch.Tensor, ...]:
    jacobian_cols: Dict[int, List[torch.Tensor]] = {}
    input = inputs[input_idx] if input is None else input
    assert input.requires_grad
    for x, idx, d_idx in _iter_tensor(input):
        wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, x)
        input_to_perturb = x[idx]
        nbhd_checks_fn = functools.partial(_check_outputs_same_dtype_and_shape, idx=idx, eps=eps)
        jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
        jacobian_cols[d_idx] = _compute_numerical_jvps_wrt_specific_input(jvp_fn, eps, x.is_complex(), is_forward_ad)
    return _combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())
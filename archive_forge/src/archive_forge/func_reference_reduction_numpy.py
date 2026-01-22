import collections
import warnings
from functools import partial, wraps
from typing import Sequence
import numpy as np
import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict
def reference_reduction_numpy(f, supports_keepdims=True):
    """Wraps a NumPy reduction operator.

    The wrapper function will forward dim, keepdim, mask, and identity
    kwargs to the wrapped function as the NumPy equivalent axis,
    keepdims, where, and initiak kwargs, respectively.

    Args:
        f: NumPy reduction operator to wrap
        supports_keepdims (bool, optional): Whether the NumPy operator accepts
            keepdims parameter. If it does not, the wrapper will manually unsqueeze
            the reduced dimensions if it was called with keepdim=True. Defaults to True.

    Returns:
        Wrapped function

    """

    @wraps(f)
    def wrapper(x: np.ndarray, *args, **kwargs):
        keys = set(kwargs.keys())
        dim = kwargs.pop('dim', None)
        keepdim = kwargs.pop('keepdim', False)
        if 'dim' in keys:
            dim = tuple(dim) if isinstance(dim, Sequence) else dim
            if x.ndim == 0 and dim in {0, -1, (0,), (-1,)}:
                kwargs['axis'] = None
            else:
                kwargs['axis'] = dim
        if 'keepdim' in keys and supports_keepdims:
            kwargs['keepdims'] = keepdim
        if 'mask' in keys:
            mask = kwargs.pop('mask')
            if mask is not None:
                assert mask.layout == torch.strided
                kwargs['where'] = mask.cpu().numpy()
        if 'identity' in keys:
            identity = kwargs.pop('identity')
            if identity is not None:
                if identity.dtype is torch.bfloat16:
                    identity = identity.cpu().to(torch.float32)
                else:
                    identity = identity.cpu()
                kwargs['initial'] = identity.numpy()
        result = f(x, *args, **kwargs)
        if keepdim and (not supports_keepdims) and (x.ndim > 0):
            dim = list(range(x.ndim)) if dim is None else dim
            result = np.expand_dims(result, dim)
        return result
    return wrapper
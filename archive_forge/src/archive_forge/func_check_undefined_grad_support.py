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
def check_undefined_grad_support(output_to_check):
    grads_output = [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in output_to_check]
    try:
        grads_input = torch.autograd.grad(output_to_check, diff_input_list, grads_output, allow_unused=True)
    except RuntimeError as e:
        warn_bc_breaking()
        raise GradcheckError('Expected backward function to handle undefined output grads. Please look at "Notes about undefined output gradients" in "tools/autograd/derivatives.yaml"') from e
    for gi, i in zip(grads_input, diff_input_list):
        if gi is not None and (not gi.eq(0).all()):
            warn_bc_breaking()
            raise GradcheckError('Expected all input grads to be undefined or zero when all output grads are undefined or zero. Please look at "Notes about undefined output gradients" in "tools/autograd/derivatives.yaml"')
    return True
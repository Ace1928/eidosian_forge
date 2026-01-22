import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any, Tuple
def warn_if_has_hooks(tensor):
    if tensor._backward_hooks:
        for k in tensor._backward_hooks:
            hook = tensor._backward_hooks[k]
            if not hasattr(k, '__torch_unserializable__'):
                warnings.warn(f'backward hook {repr(hook)} on tensor will not be serialized.  If this is expected, you can decorate the function with @torch.utils.hooks.unserializable_hook to suppress this warning')
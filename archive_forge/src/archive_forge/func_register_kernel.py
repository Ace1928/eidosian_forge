import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import torch
from torchvision import tv_tensors
def register_kernel(functional, tv_tensor_cls):
    """[BETA] Decorate a kernel to register it for a functional and a (custom) tv_tensor type.

    See :ref:`sphx_glr_auto_examples_transforms_plot_custom_tv_tensors.py` for usage
    details.
    """
    if isinstance(functional, str):
        functional = _name_to_functional(name=functional)
    elif not (callable(functional) and getattr(functional, '__module__', '').startswith('torchvision.transforms.v2.functional')):
        raise ValueError(f'Kernels can only be registered on functionals from the torchvision.transforms.v2.functional namespace, but got {functional}.')
    if not (isinstance(tv_tensor_cls, type) and issubclass(tv_tensor_cls, tv_tensors.TVTensor)):
        raise ValueError(f'Kernels can only be registered for subclasses of torchvision.tv_tensors.TVTensor, but got {tv_tensor_cls}.')
    if tv_tensor_cls in _BUILTIN_DATAPOINT_TYPES:
        raise ValueError(f'Kernels cannot be registered for the builtin tv_tensor classes, but got {tv_tensor_cls}')
    return _register_kernel_internal(functional, tv_tensor_cls, tv_tensor_wrapper=False)
from collections import OrderedDict, namedtuple
import itertools
import warnings
import functools
import weakref
import torch
from torch._prims_common import DeviceLikeType
from ..parameter import Parameter
import torch.utils.hooks as hooks
from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from typing_extensions import Self
from ...utils.hooks import RemovableHandle
def register_module_full_backward_pre_hook(hook: Callable[['Module', _grad_t], Union[None, _grad_t]]) -> RemovableHandle:
    """Register a backward pre-hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time the gradients for the module are computed.
    The hook should have the following signature::

        hook(module, grad_output) -> Tensor or None

    The :attr:`grad_output` is a tuple. The hook should
    not modify its arguments, but it can optionally return a new gradient with
    respect to the output that will be used in place of :attr:`grad_output` in
    subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
    all non-Tensor arguments.

    For technical reasons, when this hook is applied to a Module, its forward function will
    receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
    of each Tensor returned by the Module's forward function.

    Global hooks are called before hooks registered with `register_backward_pre_hook`

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    handle = hooks.RemovableHandle(_global_backward_pre_hooks)
    _global_backward_pre_hooks[handle.id] = hook
    return handle
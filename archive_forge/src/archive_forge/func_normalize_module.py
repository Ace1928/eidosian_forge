import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload
@compatibility(is_backward_compatible=False)
def normalize_module(root: torch.nn.Module, target: str, args: Tuple[Any], kwargs: Optional[Dict[str, Any]]=None, normalize_to_only_use_kwargs: bool=False) -> Optional[ArgsKwargsPair]:
    """
    Returns normalized arguments to PyTorch modules. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs).

    Args:
        root (nn.Module): root module upon which we query modules
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
    try:
        submod = root.get_submodule(target)
    except AttributeError as e:
        raise RuntimeError(f'Tried to normalize node with target {target} but root did not have that target!') from e
    if hasattr(submod.__class__, '__name__'):
        classname = submod.__class__.__name__
        if getattr(torch.nn, classname, None) == submod.__class__:
            sig = inspect.signature(inspect.unwrap(submod.forward))
            if kwargs is None:
                kwargs = {}
            new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(sig, args, kwargs, normalize_to_only_use_kwargs)
            return new_args_and_kwargs
    return None
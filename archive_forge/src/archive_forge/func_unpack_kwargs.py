from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
def unpack_kwargs(kwarg_keys: Tuple[str, ...], flat_args: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """See pack_kwargs."""
    assert len(kwarg_keys) <= len(flat_args), f'too many keys {len(kwarg_keys)} vs. {len(flat_args)}'
    if len(kwarg_keys) == 0:
        return (flat_args, {})
    args = flat_args[:-len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys):])}
    return (args, kwargs)
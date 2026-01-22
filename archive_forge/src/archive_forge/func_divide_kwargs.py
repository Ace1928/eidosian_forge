import inspect
import logging
import torch
from torch._ops import HigherOrderOperator
from torch.utils.checkpoint import checkpoint, uid
import torch._dynamo.config
@staticmethod
def divide_kwargs(kwargs):
    """
        checkpoint fn can have mixed kwargs between checkpointed fn and
        checkpoint fn itself. For example
        >> def gn(x, y, z=None):
        >>     a = torch.matmul(x, y)
        >>     if z is not None:
        >>         return torch.matmul(a, z)
        >>     return a
        >> def fn(x, y, z):
        >>     return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        In the above case, z belongs to checkpointed function gn, but
        use_reentrant belongs to the checkpoint function. This function splits
        the kwargs into checkpoint_kwargs and gmod_kwargs (or
        checkpointed_fn_kwargs).
        We do sorting to ensure same graph from run to run for better
        debuggability. It is not required for correctness.
        """
    ckpt_signature = inspect.signature(checkpoint)
    checkpoint_keys = set()
    for name in ckpt_signature.parameters:
        if name in ('function', 'args', 'kwargs'):
            continue
        checkpoint_keys.add(name)
    checkpoint_keys.add('preserve_rng_state')
    checkpoint_kwargs = {name: kwargs[name] for name in kwargs.keys() if name in checkpoint_keys}
    gmod_kwargs = {name: kwargs[name] for name in kwargs.keys() if name not in checkpoint_keys}
    return (checkpoint_kwargs, gmod_kwargs)
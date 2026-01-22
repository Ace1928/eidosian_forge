import torch
import torch.utils._pytree as pytree
from collections import namedtuple
import functools
def unpack_saved(ctx):
    flat_stuff = [None] * ctx.num_elts
    for tensor, idx in zip(ctx.saved_tensors, ctx.tensor_idxs):
        flat_stuff[idx] = tensor
    for non_tensor, idx in zip(ctx.saved_non_tensors, ctx.non_tensor_idxs):
        flat_stuff[idx] = non_tensor
    stuff = pytree.tree_unflatten(flat_stuff, ctx.spec)
    return stuff
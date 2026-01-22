import torch
import torch.utils._pytree as pytree
from collections import namedtuple
import functools
def save_pytree_for_backward(ctx, stuff):
    flat_stuff, spec = pytree.tree_flatten(stuff)
    num_elts = len(flat_stuff)
    tensor_idxs = [idx for idx, thing in enumerate(flat_stuff) if isinstance(thing, torch.Tensor)]
    non_tensor_idxs = [idx for idx, thing in enumerate(flat_stuff) if not isinstance(thing, torch.Tensor)]
    tensors = [thing for thing in flat_stuff if isinstance(thing, torch.Tensor)]
    non_tensors = [thing for thing in flat_stuff if not isinstance(thing, torch.Tensor)]
    ctx.spec = spec
    ctx.num_elts = num_elts
    ctx.save_for_backward(*tensors)
    ctx.tensor_idxs = tensor_idxs
    ctx.saved_non_tensors = non_tensors
    ctx.non_tensor_idxs = non_tensor_idxs
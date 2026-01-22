import inspect
import logging
import numpy as np
import torch
import torch.utils._pytree as pytree
import pennylane as qml
def pytreeify(cls):
    """Pytrees refer to a tree-like structure built out of container-like Python objects. The pytreeify class is used
    to bypass some PyTorch limitation of `autograd.Function`. The forward pass can only return tuple of tensors but
    not any other nested structure. This class apply flatten to the forward pass and unflatten the results in the
    apply function. In this way, it is possible to treat multiple tapes with multiple measurements.
    """
    orig_fw = cls.forward
    orig_bw = cls.backward
    orig_apply = cls.apply

    def new_apply(*inp):
        out_struct_holder = []
        flat_out = orig_apply(out_struct_holder, *inp)
        return pytree.tree_unflatten(flat_out, out_struct_holder[0])

    def new_forward(ctx, out_struct_holder, *inp):
        out = orig_fw(ctx, *inp)
        flat_out, out_struct = pytree.tree_flatten(out)
        ctx._out_struct = out_struct
        out_struct_holder.append(out_struct)
        return tuple(flat_out)

    def new_backward(ctx, *flat_grad_outputs):
        grad_outputs = pytree.tree_unflatten(flat_grad_outputs, ctx._out_struct)
        grad_inputs = orig_bw(ctx, *grad_outputs)
        return (None,) + tuple(grad_inputs)
    cls.apply = new_apply
    cls.forward = new_forward
    cls.backward = new_backward
    return cls
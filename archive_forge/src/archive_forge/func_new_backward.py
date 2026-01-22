import inspect
import logging
import numpy as np
import torch
import torch.utils._pytree as pytree
import pennylane as qml
def new_backward(ctx, *flat_grad_outputs):
    grad_outputs = pytree.tree_unflatten(flat_grad_outputs, ctx._out_struct)
    grad_inputs = orig_bw(ctx, *grad_outputs)
    return (None,) + tuple(grad_inputs)
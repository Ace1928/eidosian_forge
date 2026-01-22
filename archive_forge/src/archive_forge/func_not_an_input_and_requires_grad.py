import contextlib
import torch
import torch.utils._pytree as pytree
def not_an_input_and_requires_grad(tensor):
    if not tensor.requires_grad:
        return False
    if id(tensor) in inp_ids:
        return False
    return True
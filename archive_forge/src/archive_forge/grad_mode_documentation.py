from typing import Any, Optional
import torch
from torch.utils._contextlib import (
DO NOT USE THIS UNLESS YOU KNOW EXACTLY WHAT YOU'RE DOING.

    This context manager can lead to arbitrary silent-correctness issues in any other part of your code
    (even the ones not touched directly by the context manager)!

    Ordinarily, autograd will track mutations to tensors by incrementing it's `._version` attribute.
    This is generally important for correctness, as for example, mutating a tensor that autograd has saved
    for the backwards pass can result in incorrect gradients, and autograd uses the version counter to detect
    and error out in this situation.

    However, there are rare instances where it might be useful to hide mutations from autograd. For example:
    if a tensor is very large, and you'd like to free its memory by storing it elsewhere, and re-populate
    the tensor right before it is needed by autograd.

    Args:
        tensor (torch.Tensor): the tensor in question, that you would like to preserve the version counter of.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    
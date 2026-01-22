import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
def set_num_gradients_to_accumulate(self, num_gradients_to_accumulate: int, update_smoothing: bool=True) -> None:
    """Set the number of gradients to accumulate to a new value.

        This is experimental. This could be called while training so that
        we can gradually increasing the steps between updates. Almost always,
        `set_scale` needs to be called to update the scale as well.

        TODO (min): need a way of determine how much to increase the step size?

        TODO (min): have both `set_scale` and `set_num_gradients_to_accumulate`
        is hard to use and easy to make mistake. I think it is better
        to specific a specify a `base_scale`. But more discussion is
        needed here.

        Args:
            num_gradients_to_accumulate (int):
                Number of gradients to accumulate (calls to backward) between
                each optimizer step
            update_smoothing (bool):
                Whether to update smoothing factor or not. Default: True.
        """
    assert self._local_grad_sqr is None, "Don't change num_grad_to_accum in backward"
    assert num_gradients_to_accumulate >= 1, f'Invalid value {num_gradients_to_accumulate}'
    self._num_grads_to_accum = num_gradients_to_accumulate
    if update_smoothing:
        self._smoothing = max(1 - self._world_size * self._num_grads_to_accum / 1000, 0)
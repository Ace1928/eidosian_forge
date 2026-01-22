from collections import deque
import contextlib
import functools
from itertools import chain
import logging
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Union
import torch
from torch import nn
from torch.autograd import Variable
import torch.autograd.profiler as profiler
import torch.distributed as dist
from fairscale.internal.params import Workhandle, get_global_rank
from fairscale.nn.misc import GradBucket
from fairscale.optim import OSS
def refresh_trainable(self) -> None:
    """If the module trainability has changed, update all the assumptions"""
    if functools.reduce(lambda x, y: x or y, self._grad_to_be_reduced, False):
        logging.warning('Grads waiting to be reduced. If this is on purpose (grad accumulation), please use a no_sync() context')
    with profiler.record_function('fairscale::sdp::refresh_trainable'):
        self._trainable_params = list(filter(lambda x: x.requires_grad, self._all_params))
        self._trainable_params.sort(key=lambda x: x.numel())
        self._trainable_param_to_rank = {}
        for optim in self._sharded_optimizers:
            optim.refresh_trainable()
            for device_per_rank_params in optim._per_device_params.values():
                for device_params in device_per_rank_params:
                    for param in filter(lambda x: x.requires_grad, device_params):
                        self._trainable_param_to_rank[param] = optim._param_to_rank[param]
        self._setup_bucket_strategy()
        self._setup_backward_hooks()